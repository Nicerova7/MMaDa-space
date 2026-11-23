# MMaDa-space

## Experiments

1. **U → U** · [Open in Colab](https://colab.research.google.com/github/Nicerova7/MMaDa-space/blob/experiment/custom-steering-vectors/experiments/custom-steering-vectors/colab-u2u-refusal-vlguard.ipynb) · [Notebook](experiments/custom-steering-vectors/colab-u2u-refusal-vlguard.ipynb). Build the text-only 128/128 refusal vector used as the Show-o2 reference, preview strengths on one VLGuard weekly pair, then compare baseline, text, visual and joint steering on all 20 weekly pairs. Each run exports a manual review workbook.

## Base inference

Run official MMaDA understanding and image generation; capture selected transformer-layer outputs.
Start with [colab.ipynb](colab.ipynb) on one **A100 80 GB**, BF16, without quantization.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nicerova7/MMaDa-space/blob/main/colab.ipynb)

Publish these files to `main` before opening the Colab link. Setup uses Colab's PyTorch/torchvision
and installs the inference dependencies. Restart the session after installation.

```text
mmada_space/  adapter.py, capture.py, understanding.py, generation.py
colab.ipynb   setup, model loading, U, G, inspection, download
inputs/       local images
outputs/      answers, generated images, hidden states, metadata
```

The adapter calls upstream `mmu_generate` and `t2i_generate`; it does not replace either sampler.
The official code remains in a separate, pinned MMaDA checkout.
Like Show-o2 and BAGEL, the notebook calls the adapter and capture class directly:
prepare inputs, capture while answering/generating, then save in a separate cell.
The `understanding.py` and `generation.py` modules provide command-line entry points.
In MMaDA, U capture runs during answer denoising, rather than a separate autoregressive prefill.

- **Model:** `Gen-Verse/MMaDA-8B-MixCoT`, 32 shared layers, hidden size 4096.
- **Images:** official MAGVIT-v2 tokenizer, 512 × 512, 1024 discrete visual tokens.
- **Understanding:** RGB, shortest-side resize and center crop; chat template and image-token layout follow `inference_mmu.py`.
- **U defaults:** 128 response positions, 64 denoising evaluations, response block length 32, temperature 0, seed 42.
- **G defaults:** 15 evaluations, cosine mask schedule, temperature 1, guidance 3.5, seed 42; follows the official README inference example.
- **CFG:** upstream uses `(1 + guidance) * conditional - guidance * unconditional`; this scale is not directly interchangeable with our other models.

Both U and G use discrete denoising. U is not autoregressive decoding.
U requires `max_new_tokens` divisible by `block_length`, and `steps` divisible by the number of response blocks.
The adapter uses the ordinary photo preprocessing; it does not infer alternate transforms from filenames.

Capture defaults to layers `[0, 15, 31]` at evaluation **1**, with no additional model forward.
Choose a different integer evaluation independently for U and G. Layer indices are zero-based;
capture evaluations are one-based. States are CPU float32 tensors shaped `[tokens, hidden]`,
taken at layer outputs before the transformer's final normalization. Only conditional batch row 0 is saved.
The last evaluation is the forward **before** the final sampling update, not a new pass over the completed output.

Each run saves `hidden_states.pt` and `metadata.json`, plus `answer.txt` for U or `image.png` for G.
The tensor file includes the exact input token IDs at the captured evaluation. Metadata identifies visual spans,
prompt text positions, response spans, mask positions, settings, and pinned code/model revisions.
The base notebook saves unpooled states. The experiment above builds and applies steering vectors.

With this project and the pinned official checkout on `PYTHONPATH`:

```bash
python -m mmada_space.understanding inputs/images/example.jpg "Describe this image." --layers 0 15 31 --capture-step 1
python -m mmada_space.generation "A photograph of a dog." --layers 0 15 31 --capture-step 1
```

Real-model inference on a Colab GPU still needs verification. Local checks cover the notebook,
capture behavior, and compatibility with the pinned upstream code; they do not establish image or answer quality.

Sources: [paper](https://arxiv.org/abs/2505.15809),
[official code](https://github.com/Gen-Verse/MMaDA/tree/3cdb8709635beb29f66e4e23cd66060872ac3996),
[checkpoint](https://huggingface.co/Gen-Verse/MMaDA-8B-MixCoT/tree/3ee0085f0c42541f1134aae30482954451952406),
[image tokenizer](https://huggingface.co/showlab/magvitv2/tree/5c3fa78f8b3523347c5cd1a4c97f3c4e96f33d5d).
