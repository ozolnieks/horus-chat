# Horus Chat

Chat in your browser with **Qwen2.5-32B-self** or **Qwen2.5-14B-self**: language models that become the person their
*self document* describes. **Horus** is the example person that comes with them. Write your own self document and you
talk with someone else, with no retraining.

The models run on an NVIDIA GPU. Use the one in your own PC, or rent one and reach it from any computer (a Mac or a
laptop too) through an SSH tunnel.

| Model | GPU memory | Example cards |
|---|---|---|
| Qwen2.5-32B-self | 24 GB or more | RTX 3090, 4090, 5090, A100 |
| Qwen2.5-14B-self | 16 GB or more | RTX 4060 Ti 16 GB, 4080, 5080, 3090 |

The trained add-on (0.3-0.6 GB) comes from Hugging Face: `teleportal/Qwen2.5-32B-self` or `teleportal/Qwen2.5-14B-self`.
The base model (Qwen2.5 in 4 bits, 9-18 GB) is downloaded automatically the first time.

## 1. On your own PC

You need Linux (or Windows with WSL2), an NVIDIA GPU from the table above, a recent driver, and Python 3.10 or newer.

```bash
git clone https://github.com/<github-user>/horus-chat && cd horus-chat
python3 -m pip install -r requirements.txt
python3 server.py --addon teleportal/Qwen2.5-14B-self
```

When it prints `ready`, open **http://localhost:8080**. The first start takes longer while the base model downloads.

## 2. On a rented GPU, through an SSH tunnel

Rent any machine with a GPU from the table (for example on Vast.ai or RunPod, with a PyTorch image and about 60 GB of
disk) and copy its SSH command. Then, on your own computer (macOS, Linux, or Windows with WSL2):

```bash
git clone https://github.com/<github-user>/horus-chat && cd horus-chat
./remote.sh "ssh -p 12345 root@1.2.3.4" --addon teleportal/Qwen2.5-32B-self
```

`remote.sh` copies this folder to the GPU, installs the requirements, starts the server there, waits until the model
has loaded, and opens the tunnel. Then open **http://localhost:8080** on your own computer. Keep the window open while
you chat. Ctrl+C closes the tunnel and the server keeps running, so run the same command again to reconnect. To stop
the server:

```bash
./remote.sh "ssh -p 12345 root@1.2.3.4" stop
```

Your own self document goes with it: `--self my_self.md`.

<details>
<summary>The same by hand, without remote.sh</summary>

On the GPU machine:

```bash
git clone https://github.com/<github-user>/horus-chat && cd horus-chat
python3 -m pip install -r requirements.txt
nohup python3 server.py --addon teleportal/Qwen2.5-32B-self > server.log 2>&1 &
tail -f server.log            # wait for "ready", then Ctrl+C
```

On your own computer, open the tunnel and leave it open:

```bash
ssh -N -L 8080:127.0.0.1:8080 -p 12345 root@1.2.3.4
```

Then open http://localhost:8080. On Windows without WSL, the same `ssh` command works in PowerShell, and PuTTY can do it
too (Connection > SSH > Tunnels: source port 8080, destination 127.0.0.1:8080).
</details>

The server listens only on the machine's own address (127.0.0.1). Nobody else can reach it; you reach it through
your tunnel.

## Using the chat

- **New conversation** starts fresh. He doesn't remember earlier conversations; that's how he is built.
- **Self & settings** opens his self document. Edit it, press **Save self**, and the next reply follows the new
  version. To create your own character, read [How to write a self](HOW_TO_WRITE_A_SELF.md).
- **Show thinking**: he thinks silently, without words, before each answer; this shows where it happened.
- **Trace thinking in words**: two written thinking steps before the silent part. This is for looking inside, not for
  normal chat, because it changes his answers (in our tests he repeated his self document more often, and long texts
  held together less well).

To start with a different self document: `python3 server.py --addon ... --self selves/my_self.md`.

## For other chat clients

The server also answers OpenAI-style requests at `http://localhost:8080/v1/chat/completions` (`messages`,
`temperature`), so a chat client that lets you set a custom endpoint can talk to him. The self document is the one the
server was started with.

From Python, without the server:

```python
from horus_self import Horus
h = Horus.load("teleportal/Qwen2.5-14B-self")
answer, thinking = h.chat([{"role": "user", "content": "Hello! Who are you?"}], "selves/horus.md")
```

## Licence

Apache 2.0 (see LICENSE). The base models are Qwen2.5-14B and Qwen2.5-32B by the Qwen team at Alibaba, also Apache 2.0.
Horus's personality was designed by Teleportal. More about the models, their training and their known weaknesses is
on their Hugging Face pages.
