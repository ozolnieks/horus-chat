"""The self document: a plain-text file that says who he is.

Reading rules (the same as in training):
  - HTML comments (<!-- ... -->) are notes for people and are removed;
  - headings (lines starting with "#") and empty lines are left out;
  - every other line is one statement of the self, kept in the document's order.

The whole self is read (not a selection of lines). It is put in front of the conversation under the heading
"Who I am - I, <Name>, in my own words:" and is also read by the self channel (self_channel.py).
"""
import os
import re

from .frame import owner_from


class SelfDocument:
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.mtime = os.path.getmtime(self.path)
        with open(self.path, encoding="utf-8") as f:
            text = re.sub(r"<!--.*?-->", "", f.read(), flags=re.S)
        # (line number in the file, the statement)
        self.lines = [(n, l.strip()) for n, l in enumerate(text.split("\n"), 1)
                      if l.strip() and not l.strip().startswith("#")]
        self.owner = owner_from("\n".join(l for _, l in self.lines))

    @property
    def key(self):
        """what identifies this reading of the self: the file, its modification time and its lines"""
        return (self.path, self.mtime, tuple(n for n, _ in self.lines))

    def text(self, heading):
        """the self as the model reads it: the heading, then one statement per line"""
        return heading + "".join(line + "\n" for _, line in self.lines)


_LOADED = {}


def load_self(path):
    """the self document at `path`, read again only when the file has changed"""
    p = os.path.abspath(path)
    doc = _LOADED.get(p)
    if doc is None or doc.mtime != os.path.getmtime(p):
        doc = _LOADED[p] = SelfDocument(p)
    return doc
