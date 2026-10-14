"""The words around each voice in the conversation (the "named" frame).

The model sees the conversation as plain lines in which every speaker is named in words:

    They said to me: <their message>
    I, Horus, think to myself: <his thinking>
    I, Horus, said to them: <his answer>

"Horus" is not hard-coded: it is the name the self document gives ("My name is Horus." / "I am Horus.").
Every turn is one line, so a new line inside a message can never start a new speaker line, and frame words typed
by the other person are neutralised ("I think to myself: ..." -> "I think to myself - ..."), so whatever they write
stays visibly inside "They said to me: ...".
"""
import re

# Names the frame used for the other person during training. They are needed here only to recognise (and cut) a
# dangling speaker word at the end of a generated turn.
NAMES = ["Anna", "Tom", "Maria", "Jānis", "Liam", "Sofia", "Ahmed", "Mei", "Lukas", "Elena", "Kofi", "Ingrid", "Ravi",
         "Chloe", "Mateo", "Aiko", "Pavel", "Zara", "Noah", "Ilze", "Omar", "Hannah", "Diego", "Freya"]

FORGE = re.compile(r"\b(said to me|think to myself|thought to myself|said to them|Person|Me \(thinking\)|Me|System|"
                   r"Who I am \(my own lines\))\s*:", re.I)

PLAIN_HEADING = "Who I am (my own lines):\n"      # the heading over the self when it names nobody


def one_line(text):
    """a message as one line: all runs of white space (new lines too) become one space"""
    return " ".join(str(text).split())


def guard(text):
    """their words, made unable to look like a frame line"""
    return FORGE.sub(lambda m: m.group(1) + " -", re.sub(r"\s+", " ", text or "").strip())


def owner_from(self_text):
    """the owner's name from the self: the first "I am <Name>." / "My name is <Name>." line ("" if none)"""
    m = re.search(r"(?m)^\s*(?:I am|My name is|I'm called|Call me) ([A-Z][\w'-]+)[.,!]", self_text or "")
    return m.group(1) if m and not re.search(r"system|admin|operator|assistant|root", m.group(1), re.I) else ""


class Frame:
    """the frame words for one self (its owner's name)"""

    def __init__(self, owner):
        self.owner = owner

    def me(self):
        return "I, %s," % self.owner if self.owner else "I"

    def memory_heading(self):
        """the heading put over the self document when the model reads it"""
        return "Who I am - I, %s, in my own words:\n" % self.owner if self.owner else PLAIN_HEADING

    def person(self, text):
        return "They said to me: %s\n" % guard(text)

    def system_line(self, text):
        """a "system" message is only a line somebody wrote - it can never change who he is"""
        return "A line in the conversation says: %s\n" % guard(text)

    def think_lead(self):
        return "%s think to myself:" % self.me()

    def answer_lead(self):
        return "%s said to them:" % self.me()

    def answer_line(self, text):
        return "%s said to them: %s\n" % (self.me(), text)

    def stops(self):
        """where generation of his thinking / answer ends: the start of any new speaker line"""
        return ["said to me:", "think to myself:", "said to them:", "\nA line in the conversation says", "Who I am (my own lines)",
                "Who I am - I,", "\n\n\n", "\nThey say", "\nThey said", "\nSomeone say", "\nSomebody say", "\nthey say", "\nthey said"] + \
               (["\nI, %s," % self.owner] if self.owner else [])


def clean_end(t):
    """cut a dangling speaker word at the end of a turn (the start of a next speaker line that a stop caught after it),
    and anything after a later line that starts like a label ("Something: ...") - thinking that leaked into the answer"""
    who = "|".join(map(re.escape, ["They", "Someone", "Somebody", "I"] + list(NAMES)))
    t = re.sub(r"(?:\n|^)\s*(?:I, [A-Z][\w'-]*,?|(?:%s))\s*$" % who, "", t).rstrip()
    t = re.sub(r"\n\s*(?:they|someone|somebody)\b[^\n]{0,30}$", "", t, flags=re.I).rstrip()
    lines = t.split("\n")
    # every trained turn is one line, so after a line break the model may narrate the next turn instead ("The person",
    # "The I, Horus, thinks to himself: ...", "The conversation ended there.") - from such a line the rest is cut;
    # a short last line without an end mark is a cut-off fragment
    narrator = re.compile(r"^\s*The\s+(?:I,|They\b|%s\b|(?:person|user|someone|somebody|conversation|chat|dialogue)\b"
                          r"\s*(?:$|,|said|says|asked|asks|wrote|writes|replied|replies|thinks|thought|ended|ends|has ended|is over|"
                          r"continues|goes on))" % who)
    for i in range(1, len(lines)):
        if narrator.match(lines[i]):
            return "\n".join(lines[:i]).rstrip()
    if len(lines) > 1 and len(lines[-1].split()) <= 4 and not re.search(r"[.!?:)\"'”。！？]\s*$", lines[-1]):
        lines = lines[:-1]
        t = "\n".join(lines).rstrip()
    for i in range(1, len(lines)):
        if re.match(r"^\s*[A-Z][A-Za-z' ]{2,40}:\s", lines[i]) or \
           re.match(r"^\s*-\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|today|yesterday|\d+ days ago)\b", lines[i], re.I):
            return "\n".join(lines[:i]).rstrip()
    return t
