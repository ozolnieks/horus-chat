"""The trained add-on: the small parts that sit on top of the unchanged base model.

  addon.safetensors   the numbers
  addon_config.json   what they are and how they were trained to be used

Parts:
  self vectors        "self.*": the speaker markers ("I said", "they said", "I said to myself"), the "mine" marker on
                      every word of the self, their strength ("self.mark"), and the thinking places ("self.latent").
                      In the Horus training the markers were switched off (strength 0) - who speaks is said by the
                      frame's words and the self channel only - but they are applied exactly as in training.
  self channel        "channel.*": see self_channel.py
  base LoRA           "base_lora.*": a small low-rank addition to every attention and MLP projection of the base,
                      trained only on public-domain books and everyday dialogues (a general ability, no personality).
"""
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

FORMAT = "horus-self-addon/1"
ROLES = {"they": 0, "myself": 1}                    # "I" has its own vector (self.I)


class SelfVectors(nn.Module):
    """the speaker markers, the "mine" marker and the thinking places"""

    def __init__(self, d, latent_slots, emb_norm):
        super().__init__()
        self.I = nn.Parameter(torch.zeros(d))                     # "I said" (his answer)
        self.roles = nn.Parameter(torch.zeros(2, d))              # "they said", "I said to myself"
        self.mine = nn.Parameter(torch.zeros(d))                  # on every word of his self
        self.mark = nn.Parameter(torch.zeros(()))                 # how strongly the markers are added
        self.latent = nn.Parameter(torch.zeros(latent_slots, d))  # the thinking places (no words)
        self.emb_norm = emb_norm                                  # the base's mean word-vector size

    def marker(self, role):
        """the vector added to every word of a speaker: role is "I", "they" or "myself" """
        v = self.I if role == "I" else self.roles[ROLES[role]]
        return self.mark * self.emb_norm * v / (v.norm() + 1e-6)

    def mine_marker(self):
        return self.mark * self.emb_norm * self.mine / (self.mine.norm() + 1e-6)


class BaseLoRA(nn.Module):
    """y = W x + scale * B A x   (W: the frozen base projection; A, B: the trained low-rank addition, in float32)"""

    def __init__(self, lin, r, alpha, dev):
        super().__init__()
        self.lin, self.scale = lin, alpha / r
        self.A = nn.Parameter(torch.zeros(r, lin.in_features, device=dev), requires_grad=False)
        self.B = nn.Parameter(torch.zeros(lin.out_features, r, device=dev), requires_grad=False)

    def forward(self, x):
        y = self.lin(x)
        return y + (F.linear(F.linear(x.float(), self.A), self.B) * self.scale).to(y.dtype)


def lora_name(layer, module):
    return "base_lora.layers.%d.%s" % (layer, module)


def add_base_lora(model, cfg, tensors, dev):
    """wrap the base's projections (attention + MLP, every layer) and load the trained A, B"""
    n = 0
    for i, layer in enumerate(model.model.layers):
        for parent in (layer.self_attn, layer.mlp):
            for name in cfg["targets"]:
                if hasattr(parent, name):
                    m = BaseLoRA(getattr(parent, name), cfg["r"], cfg["alpha"], dev)
                    with torch.no_grad():
                        m.A.copy_(tensors[lora_name(i, name) + ".A"])
                        m.B.copy_(tensors[lora_name(i, name) + ".B"])
                    setattr(parent, name, m)
                    n += 1
    return n


def read_addon(path):
    """-> (config dict, {name: tensor}) from a folder holding addon_config.json + addon.safetensors"""
    from safetensors.torch import load_file
    with open(os.path.join(path, "addon_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("format") != FORMAT:
        raise ValueError("%s: not a %s add-on" % (path, FORMAT))
    return cfg, load_file(os.path.join(path, "addon.safetensors"))


def make_self_vectors(cfg, tensors, model, dev):
    emb_norm = model.get_input_embeddings().weight.float().norm(dim=-1).mean().item()
    sv = SelfVectors(model.config.hidden_size, cfg["latent_slots"], emb_norm).to(dev)
    sv.load_state_dict({k[len("self."):]: v for k, v in tensors.items() if k.startswith("self.")})
    for p in sv.parameters():
        p.requires_grad_(False)
    return sv


def check_sizes(cfg, model):
    """a clear error when the add-on was trained for another base"""
    d, n = model.config.hidden_size, len(model.model.layers)
    if cfg["hidden_size"] != d or cfg["num_layers"] != n:
        raise ValueError("the add-on is for a base with %d layers of size %d; this base has %d of size %d"
                         % (cfg["num_layers"], cfg["hidden_size"], n, d))
