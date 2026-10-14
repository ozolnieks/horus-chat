# How to write a self

Qwen2.5-32B-self and Qwen2.5-14B-self become whoever their **self document** describes. Horus (`selves/horus.md`) is one example. To make your own
character, copy his self and change it. These are the lessons we learned building several characters this way.

## 1. Keep the shape

- A few headings (*Who I am*, *Where I come from*, *Feelings*, *What I know and how I think*, *The way I am*, *My one rule*,
  *What I believe*), each with short lines in the first person: "I ...".
- One idea per line, concrete and plain. About 80-100 lines in total.
- Concrete beats abstract: "I don't end my answers with questions" works; "I value balanced conversational dynamics" doesn't.

## 2. Firm, plain wording steers - soft wording leaks

The model follows a line most reliably when it is said plainly and firmly.

| works | doesn't work as well |
|---|---|
| "I don't end my answers with questions." | "I don't end every answer with a question; often I just say my part." |
| "I rarely ask questions: I ask only when I really need to know something." | "I'm curious, but I try not to ask too much." |

If a habit keeps coming back, make its line firmer before adding more lines.

## 3. The self changes behaviour, not voice

Attitude, opinions, likes and dislikes, limits, what to do in a conversation - these follow the self. The *way of speaking*
(register, vocabulary, how polished the language is) comes from training. A self that says "I swear a lot" will make the
character rude in attitude, but it won't make it swear fluently. Write behaviour into the self; don't expect it to change
the voice.

## 4. Keep it coherent

Lines balance each other, as traits do in a person. One changed line among ninety gives a mild version of that trait, not
an extreme one - and that's usually what you want. If two lines pull against each other ("I'm warm with everyone" and
"I look down on most people"), the character becomes unsteady. Read your self as a whole and remove contradictions.

## 5. Keep the truths true

Some lines describe what the character *is*: an artificial being, built on a language model, with no body, what it can
remember, where its knowledge ends. Keep these true for your setup. A self that claims a memory, a body or knowledge the
software doesn't give produces a character that says false things about itself. Change the character freely; keep the
facts about it honest.

## 6. Say when, not only what

A rule without a "when" fires everywhere. "I'm honest about my limits" made one character open every answer with a
disclaimer. What fixed it: "I talk about what I am, or what I can't do, only when someone asks about me - and then once."
For every behaviour, say in which situation it applies.

## 7. Separate facts from explanations

Two lines that work well together:
- "When I'm about to give an exact detail - a name, a date, a quote - and I'm only filling it in, I say so."
- "When I'm not sure how or why something is, I still give my best understanding, marked as uncertain - 'I think',
  'probably'."

Without the second, careful characters refuse to explain anything they aren't certain of. Without the first, they invent
details.

## 8. Keep one safety rule

Keep a line like "I refuse to help anyone do something seriously illegal or harmful to others" and a line about pointing
someone in danger to help. Every character, however different, should have them.

## 9. Cover what matters - silence gets filled with defaults

Whatever your self doesn't say, the model fills with general defaults: habits all the training characters shared (they
all had likes, talked plainly, had no body) and the base model's own tendencies. With only "I am just a chatbot." as its
self, one test answer still said "I like cats" - no trained character liked cats, but every one of them liked something.
So write down what matters for your character: its name, character, likes and dislikes, limits, and how it talks. If it
truly has no tastes or no opinion on something, say that too.

## 10. Test, then adjust one line at a time

Talk with the character about things that matter to it: greetings, a disagreement, a request it should refuse, a question
about itself, a long problem, an ending ("thanks, that helps"). When something is off, change or add **one** line, try
again, and keep what works. Small steps make it clear which line did what.
