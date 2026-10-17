"""The model server: runs where the GPU is and serves the chat.

Usually app.py starts it for you - on this computer or on a rented GPU. By hand:
    python server.py --addon teleportal/Qwen2.5-14B-self            # then open http://localhost:8765

It listens on 127.0.0.1 only: nobody else on the network can reach it, only you (through an SSH tunnel when the GPU is
rented). The page and the status are served from the first second, while the model is still downloading and loading.

    GET  /                      the chat page (web/index.html)
    GET  /api/status            stage (loading / ready / error), error, downloaded bytes, GPU memory, the last log lines
    GET  /api/info              which model and self document are loaded
    POST /api/chat              {"messages": [...], "temperature": 0.6, "written_thinking": false, "solved_first": false}
                                -> {"answer", "thinking", "seconds"}
    GET  /api/self              the self document's text
    POST /api/self              {"text": "..."} - saves the self document; the next reply follows it
    POST /v1/chat/completions   OpenAI-style, for other chat clients: {"messages", "temperature"} -> {"choices": [...]}
"""
import argparse
import collections
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = {"h": None, "self": None, "addon": None, "stage": "starting", "error": None, "lock": threading.Lock()}
LOG = collections.deque(maxlen=200)                    # the last lines this server printed (progress bars included)


class Tee:
    """keeps what is printed (and the download / loading progress bars) for /api/status, and still prints it"""
    def __init__(self, stream):
        self.stream, self.part = stream, ""

    def write(self, s):
        self.stream.write(s)
        self.stream.flush()
        self.part += s
        while True:
            cut = min([i for i in (self.part.find("\n"), self.part.find("\r")) if i >= 0], default=-1)
            if cut < 0:
                break
            line, self.part = self.part[:cut].strip(), self.part[cut + 1:]
            if line:
                if LOG and LOG[-1].split(":")[0] == line.split(":")[0] and "%|" in line:
                    LOG[-1] = line                     # a progress bar: keep only its newest state
                else:
                    LOG.append(line)
        return len(s)

    def flush(self):
        self.stream.flush()

    def isatty(self):
        return False


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def downloaded_bytes():
    """how much of the Hugging Face cache is on disk (the base model download shows up here)"""
    try:
        from huggingface_hub import constants
        total = 0
        for d, _, files in os.walk(constants.HF_HUB_CACHE):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(d, f))
                except OSError:
                    pass
        return total
    except Exception:
        return None


def gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        name, used, total = [x.strip() for x in out.split(",")]
        return {"name": name, "used_mb": int(used), "total_mb": int(total)}
    except Exception:
        return None


def reply(messages, temperature=0.6, written_thinking=False, solved_first=False):
    """one answer from the loaded model; one at a time (the GPU holds one conversation step at a time)"""
    if STATE["stage"] != "ready":
        raise RuntimeError("the model is not ready yet (%s)" % STATE["stage"])
    msgs = [{"role": m.get("role"), "content": str(m.get("content", ""))} for m in messages
            if m.get("role") in ("user", "assistant", "system")]
    if not msgs or msgs[-1]["role"] != "user":
        raise ValueError("the last message must be from the user")
    with STATE["lock"]:
        t0 = time.time()
        answer, thinking = STATE["h"].chat(msgs, STATE["self"], temperature=float(temperature),
                                           written_thinking=bool(written_thinking), solved_first=bool(solved_first))
    return answer, thinking, round(time.time() - t0, 1)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
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
        if self.path == "/api/status":
            return self._send(200, {"stage": STATE["stage"], "error": STATE["error"], "addon": STATE["addon"],
                                    "self": os.path.basename(STATE["self"] or ""), "downloaded": downloaded_bytes(),
                                    "gpu": gpu(), "log": list(LOG)[-60:]})
        if self.path == "/api/info":
            name = os.path.basename(os.path.normpath(STATE["addon"])) if os.path.isdir(STATE["addon"]) else STATE["addon"]
            return self._send(200, {"addon": name, "self": os.path.basename(STATE["self"]),
                                    "ready": STATE["stage"] == "ready"})
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
                a, t, s = reply(j.get("messages", []), j.get("temperature", 0.6), j.get("written_thinking", False), j.get("solved_first", False))
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
                log("self document saved")
                return self._send(200, {"saved": True})
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        except RuntimeError as e:
            return self._send(503, {"error": str(e)})
        except Exception as e:                                   # a failed reply should not stop the server
            log("error in a reply: %r" % e)
            return self._send(500, {"error": "the model failed on this message: %s" % e})
        self._send(404, {"error": "not found"})

    def log_message(self, *a):
        pass


def load(addon, base):
    try:
        STATE["stage"] = "loading"
        log("loading %s - the first start downloads the base model (9-18 GB)" % addon)
        from horus_self import Horus
        STATE["h"] = Horus.load(addon, base=base)
        STATE["stage"] = "ready"
        g = gpu()
        log("ready" + (" - GPU memory %d of %d MB" % (g["used_mb"], g["total_mb"]) if g else ""))
    except BaseException as e:                                   # keep serving, so the error can be seen in the page
        STATE["stage"], STATE["error"] = "error", "%s: %s" % (type(e).__name__, e)
        log("FAILED: " + STATE["error"])
        for line in traceback.format_exc().splitlines()[-12:]:
            log("  " + line)


def main():
    ap = argparse.ArgumentParser(description="the model server for Horus Chat")
    ap.add_argument("--addon", required=True, help="Hugging Face id of the model (e.g. teleportal/Qwen2.5-14B-self) "
                                                   "or a folder with addon.safetensors + addon_config.json")
    ap.add_argument("--self", dest="self_path", default=os.path.join(HERE, "selves", "horus.md"),
                    help="the self document (default: selves/horus.md)")
    ap.add_argument("--base", default=None, help="the 4-bit base, HF id or local folder (default: the one it was trained on)")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    sys.stdout, sys.stderr = Tee(sys.stdout), Tee(sys.stderr)
    if not os.path.isfile(a.self_path):
        sys.exit("no self document at %s" % a.self_path)
    STATE.update(self=os.path.abspath(a.self_path), addon=a.addon)
    try:                                     # take the port first: a busy port should fail now, not after the download
        httpd = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    except OSError:
        sys.exit("port %d is already in use on this machine - start with another one, e.g. --port %d" % (a.port, a.port + 1))
    log("serving on http://localhost:%d (the page shows progress while the model loads)" % a.port)
    threading.Thread(target=load, args=(a.addon, a.base), daemon=True).start()
    httpd.serve_forever()


if __name__ == "__main__":
    main()
