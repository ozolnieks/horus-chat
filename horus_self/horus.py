"""Chat with the model and a self document.

What the model reads for every reply, in order:

    Who I am - I, Horus, in my own words:          <- the whole self document (also read by the self channel)
    <every statement of the self, one per line>
    They said to me: <earlier message>             <- the conversation so far, one line per turn
    I, Horus, said to them: <earlier answer>
    They said to me: <the newest message>          <- the self channel at double strength on these words
  [ I, Horus, think to myself: <compare question> <his answer>           <- only with written_thinking=True
    I, Horus, think to myself: <steps question> <his steps or "Nothing to work out."> ]
    I, Horus, think to myself: [32 thinking places]  <- trained vectors, no words: thinking without words
    I, Horus, said to them: <his answer, generated word by word>

Generation: temperature sampling (0 = always the most likely word) with a frequency penalty of 0.3 against loops;
a turn ends at the end-of-text token, at the start of any new speaker line, or after `max_new` pieces (thinking
steps: at most 110 pieces; the written steps: at most 400).

SPEED: the self is read once and kept - the channel's reading of it and the model's memory (key/value cache) of the
self text are reused for every reply while the self file is unchanged (same path, same modification time).
"""
import copy
import os
import re
import time

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from . import addon as A
from .frame import Frame, clean_end, one_line
from .model import DEFAULT_BASE, base_for, load_base
from .self_channel import SelfChannel
from .self_document import load_self

# The two written thinking steps (software questions; only his answers are generated).
COMPARE_Q = "What is this about, and does it touch anything I like, dislike or don't want to do?"
STEPS_Q = ("If this needs calculating or several steps, I work them out here one by one before I answer; "
           "if not, I write: nothing to work out.")
NOTHING_TO_WORK = "Nothing to work out."
# A message that may need working out: numbers, amounts, time, comparisons. Otherwise the steps question's answer
# "Nothing to work out." is written by software instead of generated (the same words he was trained on).
NEEDS_STEPS = re.compile(r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|hundred|"
                         r"thousand|million|half|third|quarter|double|twice|triple|percent|how many|how much|how long|how far|"
                         r"how old|calculate|count|sum|total|average|cost|price|minutes?|hours?|days?|weeks?|months?|years?|"
                         r"euros?|dollars?|km|kilomet|miles?|faster|slower|older|younger|more than|less than|times)\b", re.I)
THINK_MAX = 110          # pieces per written thinking step
STEPS_MAX = 400          # pieces for the written steps
# The optional "solved first" thinking step (2026-10-17, chat(..., solved_first=True)): for a question about making,
# reaching or solving something, he first writes in his thinking how it was most probably done, as if already done -
# then thinks silently and answers. Off by default (the answers are then exactly as without it).
SOLVED_Q = "Suppose this is already done. How was it most probably done?"
NEEDS_SOLVED = re.compile(r"\b(invent\w*|design|build)\b|\bhow\b.*\b(make|build|get|achieve|stop|solve|create|design|become|"
                          r"reach|improve|fix|turn|keep|find|grow|win|reduce|prevent|save)\b", re.I)
FREQ_PENALTY = 0.3


class Horus:
    """the base model + the trained add-on; chat() answers from any self document"""

    def __init__(self, tok, model, dev, cfg, vectors, channel):
        self.tok, self.model, self.dev, self.cfg = tok, model, dev, cfg
        self.vec, self.channel = vectors, channel
        self.msg_weight = float(cfg["channel"]["message_weight"])
        self.own_weight = float(cfg["channel"].get("own_words_weight", 1.0))
        self.times = {}                  # seconds per part of the last reply
        self._kept = {}                  # the self read once: {"key", "reading", "n", "cache"}

    @classmethod
    def load(cls, addon_dir, base=None, tiny=False):
        """addon_dir: a folder with addon.safetensors + addon_config.json, or a Hugging Face model id holding them;
        base: HF id or local folder of the 4-bit base (None = the one the add-on was trained on)"""
        if not os.path.isdir(addon_dir):
            from huggingface_hub import snapshot_download
            addon_dir = snapshot_download(addon_dir, allow_patterns=["addon/*", "addon.safetensors", "addon_config.json"])
            if os.path.isdir(os.path.join(addon_dir, "addon")):
                addon_dir = os.path.join(addon_dir, "addon")
        cfg, tensors = A.read_addon(addon_dir)
        tok, model, dev = load_base(base or base_for(cfg), tiny=tiny or cfg.get("tiny", False))
        A.check_sizes(cfg, model)
        vectors = A.make_self_vectors(cfg, tensors, model, dev)
        A.add_base_lora(model, cfg["base_lora"], tensors, dev)
        ch = cfg["channel"]
        channel = SelfChannel(model.config.hidden_size, ch["layers"], ch["r"], ch["heads"]).to(dev)
        channel.load_state_dict({k[len("channel."):]: v for k, v in tensors.items() if k.startswith("channel.")})
        for p in channel.parameters():
            p.requires_grad_(False)
        channel.attach(model)
        return cls(tok, model, dev, cfg, vectors, channel)

    # --- word vectors ---------------------------------------------------------------------------------------------
    def _emb(self):
        return self.model.get_input_embeddings()

    def words(self, text, role):
        """text as word vectors, every word marked as spoken by `role` ("I", "they", "myself") -> 1 x n x d"""
        return self._pieces([(role, text)]).unsqueeze(0)

    def _pieces(self, pieces):
        """[(role, text)] -> n x d, each piece tokenised on its own"""
        emb = self._emb()
        out = []
        for role, text in pieces:
            ids = self.tok.encode(text, add_special_tokens=False)
            out.append(emb(torch.tensor(ids, device=self.dev)) + self.vec.marker(role).to(emb.weight.dtype))
        return torch.cat(out, 0)

    def _bos(self):
        if self.tok.bos_token_id is None:            # (Qwen has none)
            return []
        return [self._emb()(torch.tensor([self.tok.bos_token_id], device=self.dev))]

    def self_embeds(self, doc, frame):
        """the self as the model reads it: heading + statements, every word with the "mine" marker"""
        emb = self._emb()
        ids = self.tok(doc.text(frame.memory_heading()), add_special_tokens=False)["input_ids"]
        return emb(torch.tensor(ids, device=self.dev)) + self.vec.mine_marker().to(emb.weight.dtype)

    def _conversation(self, messages, frame):
        pieces = []
        for m in messages:
            t = one_line(m.get("content", ""))
            if m.get("role") == "assistant":
                pieces.append(("I", frame.answer_line(t)))
            elif m.get("role") == "system":
                pieces.append(("they", frame.system_line(t)))
            else:
                pieces.append(("they", frame.person(t)))
        return self._pieces(pieces)

    # --- the reply ------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def chat(self, messages, self_path, temperature=0.6, max_new=1000, written_thinking=False, history=30, solved_first=False):
        """messages: [{"role": "user" | "assistant" | "system", "content": str}, ...] (the newest last);
        self_path: the self document; history: only the last `history` messages are read (None = all).
        written_thinking: also answer the two written thinking steps (compare, steps) before the thinking places - for
        tracing his thinking; it changes his answers (more self repetition, weaker long texts), so not for normal chat.
        solved_first: for a question about making, reaching or solving something, he first writes in his thinking how it
        was most probably done, as if already done ("Suppose this is already done..."), then answers. Off by default.
        -> (answer, thinking)"""
        model, tok, dev, emb = self.model, self.tok, self.dev, self._emb()
        if history:
            messages = list(messages)[-history:]
        if not messages:
            raise ValueError("no messages")
        doc = load_self(self_path)
        if not doc.lines:
            raise ValueError("%s: the self document is empty" % self_path)
        frame = Frame(doc.owner)
        last = max([i for i, m in enumerate(messages) if m.get("role") == "assistant"], default=-1)
        newest = " ".join(one_line(m.get("content", "")) for m in messages[last + 1:])
        self.times = {}
        t0 = time.time()

        # 1. the self: read by the channel once, and its text once into the model's memory (cache)
        kept = self._kept
        if kept.get("key") != doc.key:                 # a new or changed self: read it
            se = self.self_embeds(doc, frame)
            self.channel.read(model, se)
            prefix = torch.cat(self._bos() + [se], 0).unsqueeze(0)
            cache = DynamicCache()
            self.channel.posw = None
            self._forward(cache, 0, prefix, logits=False)
            self._kept = kept = {"key": doc.key, "reading": self.channel.current, "cache": cache, "n": prefix.shape[1]}
        self.channel.current = kept["reading"]
        cache, pos = copy.deepcopy(kept["cache"]), kept["n"]

        # 2. the conversation; the newest message's words get the channel at `message_weight`
        parts = [self._conversation(messages[:last + 1], frame)] if last >= 0 else []
        msg_from = sum(p.shape[0] for p in parts)
        if messages[last + 1:]:
            parts.append(self._conversation(messages[last + 1:], frame))
        msg_to = sum(p.shape[0] for p in parts)
        rest = torch.cat(parts, 0).unsqueeze(0)
        if self.msg_weight != 1.0 or self.own_weight != 1.0:
            w = torch.ones(rest.shape[1], device=self.channel.gate.device)
            w[msg_from:msg_to] = self.msg_weight
            self.channel.posw = w
        state = {"pos": pos}

        def feed(x):
            logits = self._forward(cache, state["pos"], x)
            state["pos"] += x.shape[1]
            return logits
        feed(rest)
        self.channel.posw = None
        self.times["reading"] = time.time() - t0; t0 = time.time()

        def own_strength(n):            # his own words: the channel at `own_words_weight` (1 for Horus)
            self.channel.posw = None if self.own_weight == 1.0 else torch.full((n,), self.own_weight, device=dev)

        def speak(role, lead, stops, limit):
            lw = self.words(lead, role); own_strength(lw.shape[1])
            logits = feed(lw); out = []
            for _ in range(limit):
                l = logits.clone()
                if out and FREQ_PENALTY:
                    ids_, cnt = torch.unique(torch.tensor(out, device=dev), return_counts=True)
                    l[ids_] -= FREQ_PENALTY * cnt.float()
                l = l / max(temperature, 1e-5)
                nxt = int(torch.multinomial(F.softmax(l, -1), 1)) if temperature > 0 else int(l.argmax())
                if nxt == tok.eos_token_id:
                    break
                out.append(nxt)
                if any(s in tok.decode(out) for s in stops):
                    break
                own_strength(1)
                logits = feed((emb(torch.tensor([nxt], device=dev)) + self.vec.marker(role).to(emb.weight.dtype)).unsqueeze(0))
            t = tok.decode(out)
            for s in stops:
                t = t.split(s)[0]
            return clean_end(t.strip())

        # 3. thinking: [the written steps], then the thinking places
        written = ""
        if written_thinking:
            ans = speak("myself", frame.think_lead() + " " + COMPARE_Q, frame.stops() + ["\n"], min(max_new, THINK_MAX))
            written = COMPARE_Q + " " + ans
            feed(self.words("\n", "myself"))
        self.times["compare"] = time.time() - t0; t0 = time.time()
        if written_thinking:
            if NEEDS_STEPS.search(newest):
                st = speak("myself", frame.think_lead() + " " + STEPS_Q, frame.stops() + ["\n"], min(max_new, STEPS_MAX))
            else:
                st = NOTHING_TO_WORK
                feed(self.words(frame.think_lead() + " " + STEPS_Q + " " + st, "myself"))
            feed(self.words("\n", "myself"))
            written = (written + "\n" + STEPS_Q + " " + st).strip()
        if solved_first and NEEDS_SOLVED.search(newest):
            sv = speak("myself", frame.think_lead() + " " + SOLVED_Q, frame.stops() + ["\n"], min(max_new, STEPS_MAX))
            feed(self.words("\n", "myself"))
            written = (written + "\n" + SOLVED_Q + " " + sv).strip()
        self.times["steps"] = time.time() - t0; t0 = time.time()
        feed(self.words(frame.think_lead(), "myself"))
        feed((self.vec.latent.to(emb.weight.dtype) + self.vec.marker("myself").to(emb.weight.dtype)).unsqueeze(0))
        feed(self.words("\n", "myself"))
        thinking = "[thinking without words: %d slots]" % self.vec.latent.shape[0]
        if written:
            thinking = (written + "\n" + thinking).strip()
        self.times["thinking places"] = time.time() - t0; t0 = time.time()

        # 4. the answer
        answer = speak("I", frame.answer_lead(), frame.stops(), max_new)
        self.times["answer"] = time.time() - t0
        return answer, thinking

    def _forward(self, cache, pos, x, logits=True):
        """x (1 x n x d) after `pos` positions already in `cache` -> the next word's logits (float32)"""
        n = x.shape[1]
        out = self.model(inputs_embeds=x, past_key_values=cache, use_cache=True,
                         attention_mask=torch.ones(1, pos + n, dtype=torch.long, device=self.dev),
                         cache_position=torch.arange(pos, pos + n, device=self.dev),
                         **({} if logits else {"logits_to_keep": 1}))
        return out.logits[0, -1].float()


_DEFAULT = {}


def load(addon_dir, base=None, tiny=False):
    """load once; then horus_self.chat(...) uses this model"""
    _DEFAULT["h"] = Horus.load(addon_dir, base=base, tiny=tiny)
    return _DEFAULT["h"]


def chat(messages, self_path, temperature=0.6, max_new=1000, written_thinking=False, solved_first=False):
    """-> (answer, thinking) with the model loaded by load()"""
    if "h" not in _DEFAULT:
        raise RuntimeError("call horus_self.load(addon_dir) first")
    return _DEFAULT["h"].chat(messages, self_path, temperature=temperature, max_new=max_new, written_thinking=written_thinking,
                               solved_first=solved_first)
