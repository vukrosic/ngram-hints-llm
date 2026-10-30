# Stop n-gram hints from leaking answers or blocking reasoning in your LLM

Code for the paper in [`paper/main.pdf`](paper/main.pdf).

## Abstract

**An n-gram hint tells a language model which token (a word or a piece of a word) most often followed the last few tokens in a large text corpus. For example, after "the capital of France is" the hint is "Paris".**

**Hints cut the NanoGPT Speedrun record from 39.9 s to 6.7 s. That record measures the time to reach a fixed loss, not a lower final loss.**

**Hints can hurt in two ways. If a token's own text is in the counts, the hint is the answer during training but not at test time. And if the model leans on hints, it may never learn the task itself.**

> [!IMPORTANT]
> Leave each token's whole page out of its training hints. If the model must also learn to reason, add hints only at the output, and expect to give up part of the speed gain.

- **Hints give perfect recall of a fact stored in the same words.** The same fact in other words gets no help.
- **Hints at the model's input blocked reasoning** in a made-up language: two-step comparison fell from 100% to chance.
- **Hints at the output only kept the recall and restored comparison.** With half dropped, sums came back too.
- **Output-only hints cost part of the gain on web text.** They kept 41% and 64% of it at the two sizes tested; with half dropped they kept 7% to 33%.
- **On web text most of the gain comes from short matches of 2 and 3 tokens**, and that gain shrinks as training gets longer.

![Correct answers per task for three ways of adding hints](facts_vs_reasoning.png)

---

## Remove the token's whole page from its training hints

During training, the text being predicted is also in the corpus. If its own count stays in, the hint is the answer, and the model trusts hints too much. Removing that one count fixes this on Shakespeare. On web text pages repeat themselves, so removing every match from the same page works better.

## Add hints at the output, and drop half, if the model must reason

Hints added at the input let the model skip the task. Adding hints only at the output keeps the model's own layers working. Dropping half then teaches it that a hint can be wrong.

---

## Where it stops working

> [!WARNING]
> The reasoning tasks are a made-up language run on small models, so they show a mechanism and not its size in real models. Engram reports reasoning gains with learned tables, which are a different method.

## Is this new?

| Work | Lookup | Own text removed | Tested by task kind |
|---|---|---|---|
| [NanoGPT Speedrun record](https://brunborg.com/n-gram-hinted-lm) | next token | earlier steps only | no |
| [Engram](https://arxiv.org/abs/2601.07372) | learned table | not needed | yes, benchmarks |
| [RETRO](https://arxiv.org/abs/2112.04426) | text chunks | same document | no |
| This repo | next token | own token or page | **yes, known answer kinds** |

## Setup

Small transformers trained with Muon, each setting run once. All numbers and commands are in [RESULTS.md](RESULTS.md). The chart comes from `chart.py`.

## Contribute

Read the [paper](paper/main.pdf) and its source in [`paper/main.tex`](paper/main.tex). The Open Questions and Planned Experiments sections list what to test next. The first open question is whether output-only hints keep more of the web text gain on much larger models; `mixed.py` trains one model on web text and reasoning tasks at once.

Open an issue with your idea or a pull request with your code, command and numbers. Runs on larger models are the most useful.
