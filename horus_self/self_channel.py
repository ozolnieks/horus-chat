"""The self channel: the self read into the model's middle ("meaning") layers.

In a few chosen layers (for Horus on Qwen2.5-32B: 20, 24, 28, 32, 36, 40 of 64) a small extra attention path reads
ONLY the self. Its result is added to the layer's output through a gate (tanh of a trained number).

  1. read():  the frozen model reads the self document once; its hidden state after each chosen layer is kept
              (no channel while reading). The channel itself stores nothing about any self - change the document
              and what the channel reads changes.
  2. then, on every forward pass, each chosen layer adds  gate * attention(this layer's states -> the self reading).

The other person's newest message gets the channel at double strength (`posw`, the per-position strength).
"""
import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfChannel(nn.Module):
    def __init__(self, d, layers, r, heads):
        super().__init__()
        self.layers = list(layers)
        self.r, self.heads = r, heads
        self.q = nn.ModuleList([nn.Linear(d, r, bias=False) for _ in self.layers])
        self.k = nn.ModuleList([nn.Linear(d, r, bias=False) for _ in self.layers])
        self.v = nn.ModuleList([nn.Linear(d, r, bias=False) for _ in self.layers])
        self.o = nn.ModuleList([nn.Linear(r, d, bias=False) for _ in self.layers])
        self.gate = nn.Parameter(torch.zeros(len(self.layers)))
        self.current = None              # {layer: (S, d) the model's reading of the self after that layer}
        self.posw = None                 # optional strength per position of the current forward pass, shape (T,)
        self.active = True
        self._hooks = []

    def attach(self, model):
        for i, l in enumerate(self.layers):
            self._hooks.append(model.model.layers[l].register_forward_hook(self._hook(i, l)))

    def _hook(self, i, l):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if not self.active or self.current is None:
                return out
            s = self.current[l]
            x = F.layer_norm(h.float(), h.shape[-1:])
            y = F.layer_norm(s.float(), s.shape[-1:])
            B, T, _ = x.shape
            dh = self.r // self.heads
            q = self.q[i](x).view(B, T, self.heads, dh).transpose(1, 2)                 # B heads T dh
            k = self.k[i](y).view(-1, self.heads, dh).transpose(0, 1).unsqueeze(0)       # 1 heads S dh
            v = self.v[i](y).view(-1, self.heads, dh).transpose(0, 1).unsqueeze(0)
            a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(dh), -1) @ v          # B heads T dh
            add = torch.tanh(self.gate[i]) * self.o[i](a.transpose(1, 2).reshape(B, T, self.r))
            if self.posw is not None and self.posw.shape[0] == T:
                add = add * self.posw.view(1, T, 1).to(add.dtype)
            h2 = h + add.to(h.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        return fn

    @torch.no_grad()
    def read(self, model, self_embeds):
        """the frozen model's reading of the self (T x d word vectors): its state after each chosen layer"""
        with self.off():
            hs = model.model(inputs_embeds=self_embeds.unsqueeze(0), output_hidden_states=True, use_cache=False).hidden_states
        self.current = {l: hs[l + 1][0].detach() for l in self.layers}

    @contextlib.contextmanager
    def off(self):
        was = self.active
        self.active = False
        try:
            yield
        finally:
            self.active = was
