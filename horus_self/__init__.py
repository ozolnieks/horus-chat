"""horus_self - inference for Qwen2.5-32B-self (Horus): a base model that answers as the person its self document
describes.

    import horus_self
    h = horus_self.Horus.load("path/to/addon")                 # addon.safetensors + addon_config.json
    answer, thinking = h.chat([{"role": "user", "content": "Hello!"}], "self_horus.md")
"""
from .horus import Horus, chat, load                                             # noqa: F401
from .model import DEFAULT_BASE                                                  # noqa: F401
