"""Capture selected layer outputs at one denoising evaluation."""
import json
from pathlib import Path

import torch


class Capture:
    def __init__(self, adapter, layers, step=1):
        self.adapter = adapter
        self.layers = list(dict.fromkeys(layers))
        self.step = step
        self.states = {}
        self.input_ids = None
        self.evaluations = 0
        self.handles = []

    def __enter__(self):
        self.states.clear()
        self.input_ids = None
        self.evaluations = 0
        self.handles.append(self.adapter.model.register_forward_pre_hook(
            self._locate, with_kwargs=True,
        ))
        for layer in self.layers:
            self.handles.append(self.adapter.layers[layer].register_forward_hook(self._hook(layer)))
        return self

    def _locate(self, module, args, kwargs):
        self.evaluations += 1
        if self.evaluations == self.step:
            ids = args[0] if args else kwargs["input_ids"]
            self.input_ids = ids[0].detach().cpu().clone()

    def _hook(self, layer):
        def save(module, args, output):
            if self.evaluations == self.step:
                self.states[layer] = output[0][0].detach().float().cpu().clone()
        return save

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def save(self, output, metadata):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        metadata = dict(metadata, capture_step=self.step, evaluations=self.evaluations,
                        layers=self.layers, cfg_row=0, state_dtype="torch.float32")
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        torch.save(dict(hidden_states=self.states, input_ids=self.input_ids, metadata=metadata),
                   output / "hidden_states.pt")
