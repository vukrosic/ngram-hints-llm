# Stop n-gram hints from leaking answers or blocking reasoning in your LLM

Code for the paper in [`paper/main.pdf`](paper/main.pdf).

## Abstract

**An n-gram hint tells a language model which token (a word or a piece of a word) most often followed the last few tokens in a large text corpus. For example, after "the capital of France is" the hint is "Paris".**

**Hints cut the NanoGPT Speedrun record from 39.9 s to 6.7 s. That record measures the time to reach a fixed loss, not a lower final loss.**

**Hints can hurt in two ways. If a token's own text is in the counts, the hint is the answer during training but not at test time. And if the model leans on hints, it may never learn the task itself.**

> [!IMPORTANT]
> Leave each token's whole page out of its training hints. If the model must also learn on its own, train every step with and without hints, and add a switch on each hint that is trained to predict whether the hint is right. In our joint test this restored comparison and lowered web loss below plain hints at the longest run.

- **Hints give perfect recall of a fact stored in the same words.** The same fact in other words gets no help.
- **Hints at the model's input blocked reasoning** in a made-up language: two-step comparison fell from 100% to chance.
- **Hints at the output only kept the recall and restored comparison.** With half dropped, sums came back too.
- **Output-only hints cost part of the gain on web text.** They kept 41% and 64% of it at the two sizes tested; with half dropped they kept 7% to 33%.
- **A wider model needs fewer statements of a fact.** It reached 50% at about 45, 10, 4 and 2 statements at widths 16, 32, 64 and 192. More layers did not change this.
- **A model that always gets right hints memorizes almost nothing itself**, even a fact stated 256 times.
- **Training every step with and without hints fixes most of this.** In a joint test it kept 96% of the web gain and restored reasoning with hints off, for 47% more clock time. Reasoning with hints on stayed blocked, and common facts did not come back.
- **Wrong hints read before the answer block reasoning, not the hint at the answer.** In a small test, comparison was 99.6% with hints only at the answer word and 51.6% with hints everywhere else.
- **Teach a switch to predict whether each hint is right.** Learned from the loss alone it stayed fully on. Taught, at widths 32, 64 and 96 it raised comparison from chance to 90.5%, 100% and 100%, and beat plain hints on web loss by 0.04, 0.07 and 0.09 at 5760 steps. It was behind at 360 steps and level at 1440. Sums with hints on stayed blocked.
- **Hints help text the model itself has not seen.** On web text the gain per token fell to zero once the model had seen the context 64 times.
- **On web text most of the gain comes from short matches of 2 and 3 tokens**, and that gain shrinks as training gets longer.

![Correct answers per task for three ways of adding hints](facts_vs_reasoning.png)

---

## Remove the token's whole page from its training hints

During training, the text being predicted is also in the corpus. If its own count stays in, the hint is the answer, and the model trusts hints too much. Removing that one count fixes this on Shakespeare. On web text pages repeat themselves, so removing every match from the same page works better.

## Add hints at the output, and drop half, if the model must reason

Hints added at the input let the model skip the task. Adding hints only at the output keeps the model's own layers working. Dropping half then teaches it that a hint can be wrong.

---

## Teach a switch to shut wrong hints

Multiply each hint by a switch between 0 and 1, read from the hidden state of a pass without hints. Give the switch its own target: does the hint equal the next token? The data already holds that answer, so this adds one small loss, on top of training every step with and without hints.

---

## Where it stops working

> [!WARNING]
> The reasoning tasks are a made-up language run on small models, so they show a mechanism and not its size in real models. The switch ran once at three widths spanning 3.2 times the parameters, and it needs a pass without hints at every step. Engram reports reasoning gains with learned tables, which are a different method.

## Is this new?

| Work | Lookup | Own text removed | Tested by task kind |
|---|---|---|---|
| [NanoGPT Speedrun record](https://brunborg.com/n-gram-hinted-lm) | next token | earlier steps only | no |
| [Engram](https://arxiv.org/abs/2601.07372) | learned table | not needed | yes, benchmarks |
| [RETRO](https://arxiv.org/abs/2112.04426) | text chunks | same document | no |
| This repo | next token | own token or page | **yes, known answer kinds** |

We found no gate trained during pretraining on whether the retrieved token is right. Engram has a learned gate with no separate target that we could find, and [kNN-Adapter](https://arxiv.org/abs/2302.10879) learns a per-token mix on top of a frozen model.

## Setup

Small transformers trained with Muon, each setting run once. All numbers and commands are in [RESULTS.md](RESULTS.md). The chart comes from `chart.py`.

## Contribute

Read the [paper](paper/main.pdf) and its source in [`paper/main.tex`](paper/main.tex). The Open Questions section lists what to test next. The first thing to test is whether the taught switch still beats plain hints on a GPT-2-sized model; `mixed.py --aux 1 --right 1` trains it on web text and reasoning tasks at once.

Open an issue with your idea or a pull request with your code, command and numbers. Runs on larger models are the most useful.
