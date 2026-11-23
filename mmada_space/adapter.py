"""Thin adapter around the pinned official MMaDA samplers."""
import random
import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoTokenizer

from models import MAGVITv2, MMadaModelLM, get_mask_schedule
from training.prompting_utils import UniversalPrompting
from training.utils import image_transform

UPSTREAM_REVISION = "3cdb8709635beb29f66e4e23cd66060872ac3996"
MODEL_ID = "Gen-Verse/MMaDA-8B-MixCoT"
MODEL_REVISION = "3ee0085f0c42541f1134aae30482954451952406"
VQ_ID = "showlab/magvitv2"
VQ_REVISION = "5c3fa78f8b3523347c5cd1a4c97f3c4e96f33d5d"


class MMaDAAdapter:
    def __init__(self):
        self.device = torch.device("cuda:0")
        self.dtype = torch.bfloat16
        self.resolution = 512
        self.num_visual_tokens = 1024
        path = snapshot_download(MODEL_ID, revision=MODEL_REVISION)
        self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
        self.prompting = UniversalPrompting(
            self.tokenizer, max_text_len=512, use_reserved_token=True,
            cond_dropout_prob=0.0,
        )
        self.model = MMadaModelLM.from_pretrained(
            path, torch_dtype=self.dtype,
        ).to(self.device).eval().requires_grad_(False)
        vq_path = snapshot_download(VQ_ID, revision=VQ_REVISION)
        self.vq = MAGVITv2.from_pretrained(vq_path).to(self.device).eval().requires_grad_(False)
        self.layers = self.model.model.transformer.blocks

    def autocast(self):
        return torch.autocast("cuda", dtype=self.dtype)

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def metadata(self):
        return dict(
            model=MODEL_ID, model_revision=MODEL_REVISION,
            upstream_revision=UPSTREAM_REVISION, vq_model=VQ_ID,
            vq_revision=VQ_REVISION, dtype=str(self.dtype), resolution=self.resolution,
            capture="layer output before final norm; conditional row; [tokens, hidden]",
            use_cache=False,
        )

    @torch.no_grad()
    def prepare_understanding(self, image_path, question):
        with Image.open(image_path) as source:
            pixels = image_transform(source.convert("RGB"), resolution=self.resolution)
        # The official image tokenizer runs in FP32, outside transformer autocast.
        visual_ids = self.vq.get_code(pixels.unsqueeze(0).to(self.device))
        visual_ids = visual_ids + len(self.tokenizer)
        text_ids = self.tokenizer.apply_chat_template(
            [dict(role="user", content=question)], tokenize=True,
            add_generation_prompt=True, return_tensors="pt",
        ).to(self.device)
        special = self.prompting.sptids_dict
        prefix = torch.cat([special["<|mmu|>"], special["<|soi|>"]]).reshape(1, -1)
        end = special["<|eoi|>"].reshape(1, -1)
        return torch.cat([
            prefix.to(self.device), visual_ids, end.to(self.device), text_ids,
        ], dim=1).long()

    @torch.no_grad()
    def answer(self, input_ids, max_new_tokens=128, steps=64, block_length=32, seed=42):
        self.seed(seed)
        with self.autocast():
            output = self.model.mmu_generate(
                input_ids.clone(), max_new_tokens=max_new_tokens, steps=steps,
                block_length=block_length, temperature=0.0,
            )
        return self.tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)

    @torch.no_grad()
    def prepare_generation(self, prompt):
        masked = torch.full(
            (1, self.num_visual_tokens), self.model.config.mask_token_id,
            dtype=torch.long, device=self.device,
        )
        input_ids, attention_mask = self.prompting(([prompt], masked), "t2i_gen")
        uncond_ids, uncond_mask = self.prompting(([""], masked), "t2i_gen")
        return dict(input_ids=input_ids, attention_mask=attention_mask,
                    uncond_input_ids=uncond_ids, uncond_attention_mask=uncond_mask)

    @torch.no_grad()
    def generate(self, prompt, steps=15, guidance=3.5, seed=42):
        self.seed(seed)
        inputs = self.prepare_generation(prompt)
        with self.autocast():
            codes = self.model.t2i_generate(
                **inputs, timesteps=steps, guidance_scale=guidance,
                temperature=1.0, noise_schedule=get_mask_schedule("cosine"),
                seq_len=self.num_visual_tokens, resolution=self.resolution,
                mask_token_id=self.model.config.mask_token_id, uni_prompting=self.prompting,
            )
        pixels = self.vq.decode_code(codes.clamp(0, 8191))
        pixels = ((pixels + 1.0) / 2.0).clamp(0, 1)
        array = (pixels[0].permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)
        return Image.fromarray(array)

    def alignment(self, input_ids, prompt_length):
        ids = input_ids.tolist()
        special = self.prompting.sptids_dict
        visual_start = ids.index(int(special["<|soi|>"])) + 1
        visual_end = ids.index(int(special["<|eoi|>"]))
        special_ids = set(self.tokenizer.all_special_ids) | {int(value) for value in special.values()}
        return dict(
            visual_span=[visual_start, visual_end], prompt_length=prompt_length,
            prompt_text_positions=[i for i, token in enumerate(ids[:prompt_length])
                                   if not visual_start <= i < visual_end and token not in special_ids],
            response_span=[prompt_length, len(ids)],
            mask_positions=[i for i, token in enumerate(ids) if token == self.model.config.mask_token_id],
            span_convention="zero-based, end-exclusive; input_ids identify special/padding tokens",
        )
