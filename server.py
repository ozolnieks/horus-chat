"""Chat with a -self model (Horus, or whoever your self document describes) in the browser.

Run it where the GPU is:
    python server.py --addon teleportal/Qwen2.5-32B-self            # then open http://localhost:8080

On your own PC that is all. On a rented GPU, start it there and reach it through an SSH tunnel (remote.sh does both).
It listens on 127.0.0.1 only: nobody else on the network can reach it, only you through the tunnel.

    GET  /                      the chat page (web/index.html)
    GET  /api/info              which model and self document are loaded
    POST /api/chat              {"messages": [...], "temperature": 0.6, "written_thinking": false}
                                -> {"answer", "thinking", "seconds"}
    GET  /api/self              the self document's text
    POST /api/self              {"text": "..."} - saves the self document; the next reply follows it
    POST /v1/chat/completions   OpenAI-style, for other chat clients: {"messages", "temperature"} -> {"choices": [...]}
"""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = {"h": None, "self": None, "addon": None, "lock": threading.Lock()}


def reply(messages, temperature=0.6, written_thinking=False):
    """one answer from the loaded model; one at a time (the GPU holds one conversation step at a time)"""
    msgs = [{"role": m.get("role"), "content": str(m.get("content", ""))} for m in messages
            if m.get("role") in ("user", "assistant", "system")]
    if not msgs or msgs[-1]["role"] != "user":
        raise ValueError("the last message must be from the user")
    with STATE["lock"]:
        t0 = time.time()
        answer, thinking = STATE["h"].chat(msgs, STATE["self"], temperature=float(temperature),
                                           written_thinking=bool(written_thinking))
    return answer, thinking, round(time.time() - t0, 1)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith(("text", "application/json")) else ""))
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def _json(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "web", "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html")
        if self.path == "/api/info":
            name = os.path.basename(os.path.normpath(STATE["addon"])) if os.path.isdir(STATE["addon"]) else STATE["addon"]
            return self._send(200, {"addon": name, "self": os.path.basename(STATE["self"]),
                                    "ready": STATE["h"] is not None})
        if self.path == "/api/self":
            with open(STATE["self"], encoding="utf-8") as f:
                return self._send(200, {"text": f.read(), "name": os.path.basename(STATE["self"])})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            j = self._json()
        except ValueError:
            return self._send(400, {"error": "bad json"})
        try:
            if self.path == "/api/chat":
                a, t, s = reply(j.get("messages", []), j.get("temperature", 0.6), j.get("written_thinking", False))
                return self._send(200, {"answer": a, "thinking": t, "seconds": s})
            if self.path == "/v1/chat/completions":
                a, t, s = reply(j.get("messages", []), j.get("temperature", 0.6))
                return self._send(200, {"object": "chat.completion", "model": STATE["addon"],
                                        "choices": [{"index": 0, "finish_reason": "stop",
                                                     "message": {"role": "assistant", "content": a}}]})
            if self.path == "/api/self":
                text = str(j.get("text", ""))
                if not text.strip():
                    return self._send(400, {"error": "the self document can't be empty"})
                with open(STATE["self"], "w", encoding="utf-8") as f:
                    f.write(text if text.endswith("\n") else text + "\n")
                return self._send(200, {"saved": True})
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:                                   # a failed reply should not stop the server
            print("error:", repr(e), file=sys.stderr, flush=True)
            return self._send(500, {"error": "the model failed on this message: %s" % e})
        self._send(404, {"error": "not found"})

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser(description="chat with a -self model in the browser")
    ap.add_argument("--addon", required=True, help="Hugging Face id of the model (e.g. teleportal/Qwen2.5-32B-self) "
                                                   "or a local folder with addon.safetensors + addon_config.json")
    ap.add_argument("--self", dest="self_path", default=os.path.join(HERE, "selves", "horus.md"),
                    help="the self document (default: selves/horus.md)")
    ap.add_argument("--base", default=None, help="the 4-bit base, HF id or local folder (default: the one it was trained on)")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    if not os.path.isfile(a.self_path):
        sys.exit("no self document at %s" % a.self_path)
    STATE.update(self=os.path.abspath(a.self_path), addon=a.addon)
    print("loading %s (the first start downloads the base model, 9-18 GB) ..." % a.addon, flush=True)
    from horus_self import Horus
    STATE["h"] = Horus.load(a.addon, base=a.base)
    print("ready: open http://localhost:%d  (self: %s)" % (a.port, a.self_path), flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
