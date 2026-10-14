"""Loading the base model.

The base is Qwen2.5-32B in 4 bits (NF4, double quantisation, bfloat16 compute) - the pre-quantised copy
`unsloth/Qwen2.5-32B-bnb-4bit`, given as a Hugging Face id or a local folder. It is loaded exactly the way it was
loaded in training: as saved, on GPU 0, without asking for another dtype (transformers then keeps the parts that
are not quantised - word vectors, norms, output layer - in float16). The base is frozen; nothing in it is changed
except that the base LoRA (addon.py) is wrapped around its projections.

`tiny=True` is a test hook: a tiny random Qwen2 model (4 layers of size 64) with Qwen's real tokenizer, in float32,
built from the current torch random state. It is the same tiny model the research code builds, so the two can be
compared on a CPU (tests/test_exact.py).
"""
import torch

DEFAULT_BASE = "unsloth/Qwen2.5-32B-bnb-4bit"
TINY_SOURCE = "Qwen/Qwen2.5-0.5B-Instruct"         # tokenizer + config shape for the tiny test model


def base_for(cfg):
    """the 4-bit base an add-on was trained on: its config names the original ("Qwen/Qwen2.5-14B") and the training
    used unsloth's pre-quantised copy of it ("unsloth/Qwen2.5-14B-bnb-4bit")"""
    name = cfg.get("base_model") or ""
    if name.startswith("Qwen/"):
        return "unsloth/%s-bnb-4bit" % name.split("/", 1)[1]
    return name or DEFAULT_BASE


def load_base(base=DEFAULT_BASE, tiny=False):
    """-> (tokenizer, model, device)"""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if tiny:
        tok = AutoTokenizer.from_pretrained(TINY_SOURCE)
        cfg = AutoConfig.from_pretrained(TINY_SOURCE)
        cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size = 4, 64, 128
        cfg.num_attention_heads, cfg.num_key_value_heads = 4, 2
        model = AutoModelForCausalLM.from_config(cfg).to(torch.float32).to(dev)
    else:
        if dev != "cuda":
            raise RuntimeError("the 4-bit base needs an NVIDIA GPU (bitsandbytes): about 24 GB of GPU memory for the 32B, 16 GB for the 14B")
        tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base, device_map={"": 0})
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return tok, model, dev
