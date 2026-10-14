"""Horus Chat - start it on your own computer and open the page:

    python3 app.py                      # then open http://localhost:8765

In the page, choose where the model runs:
  - a rented GPU: paste the SSH command your provider gives you (Vast.ai, RunPod, ...). The app copies the model server
    there, installs it, starts it, opens an SSH tunnel and shows every step, the download and the GPU's log.
  - this computer: if it has a suitable NVIDIA GPU, the model server is started here.
Then you chat in the same page. Only Python's standard library is needed here; the rented machine gets the rest.
SSH uses your own keys, as your terminal does; the app never asks for or stores a password.
"""
import argparse
import collections
import io
import json
import os
import shlex
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
SHIP = ["server.py", "requirements.txt", "web", "horus_self"]     # the app files the GPU machine needs (updated each time)
SHIP_ONCE = ["selves"]                                           # copied only if missing: edits made there are kept
NEED_GB = {"14B": 16, "32B": 24}                                         # GPU memory each size needs
STALL_S = 120                                                            # no download progress this long = stalled
A = {"banner": set(), "mode": None, "phase": "idle", "error": None, "hint": None, "ssh": None, "addon": None, "port": None,
     "tunnel": None, "proc": None, "rport": None, "dl": None, "dl_t": None, "stalled": False, "gpu": None}
LOG = collections.deque(maxlen=400)
LOCK = threading.Lock()


def log(msg):
    for line in str(msg).splitlines() or [""]:
        LOG.append(time.strftime("%H:%M:%S ") + line)
    print(time.strftime("%H:%M:%S ") + str(msg), flush=True)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def fail(error, hint=None):
    A.update(phase="error", error=error, hint=hint)
    log("ERROR: " + error + (" - " + hint if hint else ""))


def size_of(addon):
    return "32B" if "32B" in addon else "14B" if "14B" in addon else None


# --- the SSH side -----------------------------------------------------------------------------------------------------
VALUE_OPTS = {"-p", "-i", "-o", "-l", "-F", "-J", "-c", "-m", "-b", "-E", "-e", "-I", "-Q", "-S", "-W", "-w", "-B", "-O"}


def ssh_base(line):
    """the user's ssh command -> our base command: their options and host, without port forwards, plus safe options"""
    try:
        parts = shlex.split(line.strip())
    except ValueError:
        raise ValueError("the SSH command can't be read - paste it exactly as the provider shows it")
    if not parts or os.path.basename(parts[0]) not in ("ssh", "ssh.exe"):
        raise ValueError('paste the SSH command, e.g. "ssh -p 12345 root@1.2.3.4"')
    keep, i = [], 1
    while i < len(parts):
        p = parts[i]
        if p in ("-L", "-R", "-D"):                    # the provider's own port forwards are not needed (and may clash)
            i += 2
            continue
        if p.startswith(("-L", "-R", "-D")) and len(p) > 2 or p in ("-N", "-f", "-t", "-T"):
            i += 1
            continue
        keep.append(p)
        i += 1
    opts, host, i = [], None, 0
    while i < len(keep):                               # options (with their values) go before the host, as ssh needs
        if keep[i] in VALUE_OPTS and i + 1 < len(keep):
            opts += keep[i:i + 2]
            i += 2
        elif keep[i].startswith("-"):
            opts.append(keep[i])
            i += 1
        else:
            host = host or keep[i]                     # (anything after the host that is not an option is dropped)
            i += 1
    if not host:
        raise ValueError("the SSH command has no host in it (like root@1.2.3.4)")
    keep = opts + [host]
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ServerAliveInterval=30"] + keep


def ssh_run(cmd, stdin=None, stream=False, timeout=None):
    """run one command on the GPU machine -> (exit code, output); stream=True logs the output line by line"""
    args = A["ssh"] + [cmd]
    if not stream:
        r = subprocess.run(args, input=stdin, capture_output=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).decode("utf-8", "replace")
    p = subprocess.Popen(args, stdin=subprocess.PIPE if stdin else None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if stdin:
        p.stdin.write(stdin)
        p.stdin.close()
    out = []
    for raw in p.stdout:
        line = raw.decode("utf-8", "replace").rstrip()
        out.append(line)
        if line and line not in A["banner"] and "WARNING: Running pip as" not in line and "Activated conda" not in line:
            log("  " + line[:300])
    return p.wait(), "\n".join(out)


def explain_ssh(out):
    o = out.lower()
    if "permission denied" in o:
        return ("the GPU machine refused the connection",
                "add your SSH public key in the provider's account settings, then try again (password logins are not supported)")
    if "could not resolve" in o or "name or service not known" in o:
        return "the host name in the SSH command can't be found", "check that you pasted the whole command"
    if "connection refused" in o or "timed out" in o or "no route" in o or "connection closed" in o:
        return "can't reach the GPU machine", "is it running? Check its status at your provider and the port in the SSH command"
    if "host key verification failed" in o or "remote host identification has changed" in o:
        return ("the machine's identity changed since your last connection",
                "a provider can reuse an address for a new machine: remove the old entry with ssh-keygen -R, then retry")
    return "the SSH connection failed", out.strip().splitlines()[-1][:200] if out.strip() else None


def pack(names):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name in names:
            t.add(os.path.join(HERE, name), arcname=name,
                  filter=lambda ti: None if "__pycache__" in ti.name or ti.name.endswith((".pyc", ".DS_Store")) else ti)
    return buf.getvalue()


START_SCRIPT = r"""
cd ~/horus-chat || exit 3
[ -f /venv/main/bin/activate ] && . /venv/main/bin/activate >/dev/null 2>&1
if [ -f server.pid ] && kill -0 "$(cat server.pid)" 2>/dev/null; then
  if [ "$(cat server.addon 2>/dev/null)" = "$ADDON" ]; then echo "RUNNING $(cat server.port)"; exit 0; fi
  kill "$(cat server.pid)"; sleep 3                       # another model is running there: replace it
fi
P=$(python3 -c 'import socket,sys; p=int(sys.argv[1]); s=socket.socket(); r=s.connect_ex(("127.0.0.1",p)); s.close(); print(p if r else 0)' 8765)
if [ "$P" = 0 ]; then P=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])'); fi
nohup python3 server.py --addon "$ADDON" --port "$P" > server.log 2>&1 < /dev/null &
echo $! > server.pid; echo "$P" > server.port; echo "$ADDON" > server.addon; echo "STARTED $P"
"""


def connect_remote(line, addon):
    try:
        A.update(mode="remote", phase="connecting", error=None, hint=None, addon=addon, stalled=False, dl=None, gpu=None)
        LOG.clear()
        A["ssh"] = ssh_base(line)
        log("connecting: " + " ".join(shlex.quote(x) for x in A["ssh"][9:]))
        code, out = ssh_run("true", timeout=60)       # prints nothing itself: whatever appears is the provider's login message
        if code != 0:
            return fail(*explain_ssh(out))
        A["banner"] = {x.rstrip() for x in out.splitlines() if x.strip()}
        code, out = ssh_run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true", timeout=60)
        g = [x for x in out.splitlines() if x.strip() and x.rstrip() not in A["banner"]]
        gline = next((x for x in g if "MiB" in x), None)
        if not gline:
            return fail("no NVIDIA GPU found on that machine", "rent a machine with an NVIDIA GPU (16 GB or more)")
        log("GPU: " + gline.strip())
        need = NEED_GB.get(size_of(addon) or "", 0)
        try:
            have = int(gline.split(",")[-1].strip().split()[0]) / 1024
            if need and have < need - 0.5:
                return fail("this GPU has %.0f GB of memory; %s needs about %d GB" % (have, size_of(addon), need),
                            "choose the 14B model, or rent a bigger GPU")
        except ValueError:
            pass

        A["phase"] = "copying"
        for names, extract in ((SHIP, "tar xzf -"), (SHIP_ONCE, "tar xzf - --skip-old-files")):
            code, out = ssh_run("mkdir -p ~/horus-chat && cd ~/horus-chat && " + extract, stdin=pack(names), timeout=300)
            if code != 0:
                return fail("copying the app to the GPU machine failed", out.strip()[-200:])
        log("app copied to ~/horus-chat on the GPU machine")

        A["phase"] = "installing"
        log("installing the requirements there (the first time takes a minute or two) ...")
        code, out = ssh_run("cd ~/horus-chat && { [ -f /venv/main/bin/activate ] && . /venv/main/bin/activate; true; } && "
                            "python3 -m pip install --progress-bar off -q -r requirements.txt", stream=True)
        if code != 0:
            return fail("installing the requirements failed", "see the log; a PyTorch image with CUDA usually has everything needed")

        A["phase"] = "starting"
        code, out = ssh_run("ADDON=%s bash -s" % shlex.quote(addon), stdin=START_SCRIPT.encode(), timeout=60)
        word = next((x for x in out.split() if x in ("RUNNING", "STARTED")), None)
        if code != 0 or not word:
            return fail("starting the model server failed", out.strip()[-200:])
        A["rport"] = int(out.split(word)[1].split()[0])
        log(("the model server was already running" if word == "RUNNING" else "model server started") +
            " (port %d on the GPU machine)" % A["rport"])
        open_tunnel()
    except ValueError as e:
        fail(str(e))
    except subprocess.TimeoutExpired:
        fail("the GPU machine stopped answering", "check its status at your provider, then connect again")
    except FileNotFoundError:
        fail("ssh was not found on this computer", "install OpenSSH (built into macOS, Linux and Windows 10/11)")


def open_tunnel():
    A["phase"] = "tunnel"
    A["port"] = free_port()
    A["tunnel"] = subprocess.Popen(A["ssh"] + ["-N", "-o", "ExitOnForwardFailure=yes", "-L",
                                               "%d:127.0.0.1:%d" % (A["port"], A["rport"])],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for _ in range(40):
        if A["tunnel"].poll() is not None:
            return fail("the SSH tunnel closed", (A["tunnel"].stderr.read() or b"").decode("utf-8", "replace").strip()[-200:])
        if backend("/api/status") is not None:
            log("tunnel open - this page now talks to the GPU machine")
            A["phase"] = "loading"
            return
        time.sleep(0.5)
    fail("the model server on the GPU machine doesn't answer", "see its log below; connect again to retry")


# --- this computer ------------------------------------------------------------------------------------------------
def start_local(addon):
    A.update(mode="local", phase="starting", error=None, hint=None, addon=addon, stalled=False, dl=None, gpu=None)
    LOG.clear()
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        out = ""
    if not out:
        return fail("no NVIDIA GPU found on this computer", "use a rented GPU instead (the other option)")
    log("GPU: " + out.splitlines()[0])
    try:
        import torch, transformers, bitsandbytes  # noqa: F401
    except ImportError as e:
        return fail("the model's Python packages are missing here (%s)" % e.name,
                    "run: python3 -m pip install -r requirements.txt   then start again")
    A["port"] = free_port()
    A["proc"] = subprocess.Popen([sys.executable, "-u", os.path.join(HERE, "server.py"), "--addon", addon,
                                  "--port", str(A["port"])], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=HERE)
    for _ in range(60):
        if A["proc"].poll() is not None:
            return fail("the model server stopped right away", "run python3 server.py --addon %s to see why" % addon)
        if backend("/api/status") is not None:
            A["phase"] = "loading"
            return log("model server started on this computer")
        time.sleep(0.5)
    fail("the model server doesn't answer")


# --- talking to the model server ------------------------------------------------------------------------------------
def backend(path, body=None, timeout=5):
    """-> parsed JSON from the model server, or None if it can't be reached"""
    if not A["port"]:
        return None
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (A["port"], path),
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except (urllib.error.URLError, OSError, ValueError):
        return None


def state():
    """everything the page shows: our phase and log, and the model server's status"""
    st = None
    if A["phase"] in ("loading", "ready") or (A["phase"] == "error" and A["port"]):
        st = backend("/api/status", timeout=8)
        if A["mode"] == "remote" and A["tunnel"] is not None and A["tunnel"].poll() is not None and A["phase"] != "error":
            fail("the connection to the GPU machine was lost", "press Connect again - the model server there keeps running")
        elif A["mode"] == "local" and A["proc"] is not None and A["proc"].poll() is not None and A["phase"] != "error":
            fail("the model server on this computer stopped", "see its log; start again")
        elif st:
            A["gpu"] = st.get("gpu")
            if st["stage"] == "ready" and A["phase"] != "ready":
                A["phase"] = "ready"
                log("the model is ready - you can chat")
            elif st["stage"] == "error" and A["phase"] != "error":
                fail("the model failed to load: " + (st.get("error") or "unknown error"), "see the model server's log")
            elif st["stage"] in ("loading", "starting"):
                d, now = st.get("downloaded"), time.time()
                if d is not None and d != A["dl"]:
                    A.update(dl=d, dl_t=now, stalled=False)
                elif A["dl_t"] and now - A["dl_t"] > STALL_S and not A["stalled"]:
                    A["stalled"] = True
                    log("the download has made no progress for %d minutes - it may have stalled" % (STALL_S // 60))
    return {"app": True, "mode": A["mode"], "phase": A["phase"], "error": A["error"], "hint": A["hint"],
            "addon": A["addon"], "stalled": A["stalled"], "log": list(LOG)[-120:], "server": st}


def stop_server():
    if A["mode"] == "remote" and A["ssh"]:
        code, out = ssh_run('cd ~/horus-chat 2>/dev/null && [ -f server.pid ] && kill "$(cat server.pid)" 2>/dev/null '
                            '&& rm -f server.pid server.port server.addon && echo stopped || echo "no server running"', timeout=30)
        log("model server on the GPU machine: " + out.strip().splitlines()[-1] if out.strip() else "stop sent")
    if A["proc"] is not None and A["proc"].poll() is None:
        A["proc"].terminate()
        log("model server on this computer stopped")
    disconnect()


def disconnect():
    if A["tunnel"] is not None and A["tunnel"].poll() is None:
        A["tunnel"].terminate()
    A.update(mode=None, phase="idle", error=None, hint=None, port=None, tunnel=None, proc=None, rport=None)


# --- the page and the proxy -----------------------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        return self.rfile.read(n) if n else b""

    def _proxy(self, body=None):
        if not A["port"]:
            return self._send(503, {"error": "not connected to a model server"})
        req = urllib.request.Request("http://127.0.0.1:%d%s" % (A["port"], self.path), data=body,
                                     headers={"Content-Type": "application/json"}, method=self.command)
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                return self._send(r.status, r.read())
        except urllib.error.HTTPError as e:
            return self._send(e.code, e.read())
        except (urllib.error.URLError, OSError):
            return self._send(502, {"error": "the model server can't be reached - see the log"})

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "web", "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html")
        if self.path == "/app/state":
            return self._send(200, state())
        if self.path.startswith(("/api/", "/v1/")):
            return self._proxy()
        self._send(404, {"error": "not found"})

    def do_POST(self):
        raw = self._body()
        if self.path.startswith(("/api/", "/v1/")):
            return self._proxy(raw)
        try:
            j = json.loads(raw or b"{}")
        except ValueError:
            return self._send(400, {"error": "bad json"})
        addon = str(j.get("addon") or "").strip()
        if self.path in ("/app/connect", "/app/local"):
            if not addon:
                return self._send(400, {"error": "choose a model"})
            if not LOCK.acquire(blocking=False):
                return self._send(409, {"error": "already connecting"})

            def run():
                try:
                    disconnect()
                    if self.path == "/app/connect":
                        connect_remote(str(j.get("ssh", "")), addon)
                    else:
                        start_local(addon)
                finally:
                    LOCK.release()
            threading.Thread(target=run, daemon=True).start()
            return self._send(200, {"started": True})
        if self.path == "/app/stop":
            threading.Thread(target=stop_server, daemon=True).start()
            return self._send(200, {"stopping": True})
        if self.path == "/app/disconnect":
            disconnect()
            return self._send(200, {"disconnected": True})
        self._send(404, {"error": "not found"})

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser(description="Horus Chat: chat with a -self model on this computer or on a rented GPU")
    ap.add_argument("--port", type=int, default=8765, help="the page's port on this computer (default 8765)")
    ap.add_argument("--no-browser", action="store_true", help="don't open the browser")
    a = ap.parse_args()
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    except OSError:
        sys.exit("port %d is in use - start with another one: python3 app.py --port %d" % (a.port, a.port + 1))
    url = "http://localhost:%d" % a.port
    print("Horus Chat is running: open %s   (Ctrl+C to quit)" % url, flush=True)
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if A["tunnel"] is not None and A["tunnel"].poll() is None:
            A["tunnel"].terminate()
        if A["proc"] is not None and A["proc"].poll() is None:
            print("stopping the model server on this computer", flush=True)
            A["proc"].terminate()


if __name__ == "__main__":
    main()
