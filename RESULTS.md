# Results

Each setting was run once. Loss is cross-entropy on the next token; lower is better.

## Facts and reasoning (`facts.py`)

A made-up language with 1.5M lines. Width 64, 4 layers, 4 heads, 5760 steps, batch 32, learning rate 0.02. Accuracy at the answer token on 1000 new lines per group. Chance is 1% for facts, about 10% for sums and 50% for read and comparison.

| Hints | Common fact | Rare fact, same words | Rare fact, other words | Not in table | Held-out sum | Read | Comparison |
|---|---|---|---|---|---|---|---|
| None | .995 | .009 | .005 | .013 | .492 | .480 | 1.000 |
| All 3 places | 1.000 | 1.000 | .010 | .004 | 0 | .502 | .495 |
| All 3 places, half dropped | 1.000 | 1.000 | .006 | .012 | 0 | .513 | .495 |
| Input only | 1.000 | 1.000 | .006 | .012 | 0 | .513 | .497 |
| Length 6 only, all 3 places | 1.000 | 1.000 | .012 | .010 | .730 | .474 | .499 |
| Output only | 1.000 | 1.000 | .010 | .003 | 0 | .924 | 1.000 |
| **Output only, half dropped** | **1.000** | **1.000** | **.006** | **.012** | **.536** | **.499** | **.999** |

We cannot explain the read score of .924 at output only. Runs took about 190 s. There is no size or length check for this test.

```bash
python3 facts.py --hints 0 --steps 5760 --lr 0.02
python3 facts.py --hints 1 --steps 5760 --lr 0.02
python3 facts.py --hints 1 --steps 5760 --lr 0.02 --sites 2 --drop 0.5
```

`--sites` picks where hints enter (0 input, 1 middle, 2 output), `--drop` the share dropped, `--use` the lengths as indices of 2, 3, 4, 6.

## Web text and reasoning in one model (`mixed.py`)

Each step trains on 2 web chunks and 32 made-up lines, for 5760 steps at learning rate 0.02. Web hints use lengths 2 to 20 and leave out the token's page. Rare facts in the table scored 1.000 in every hint row. Runs took 537 to 888 s.

| Width | Hints | Web, hints on | Web, hints off | Held-out sum | Comparison |
|---|---|---|---|---|---|
| 32 | None | - | 7.132 | .493 | .600 |
| 32 | All 3 places | 6.824 | 7.846 | 0 | .496 |
| 32 | Output only, half dropped | 7.095 | 7.107 | .458 | .929 |
| 64 | None | - | 6.991 | .350 | .971 |
| 64 | All 3 places | 6.714 | 7.637 | .048 | .496 |
| 64 | Output only | 6.878 | 7.061 | 0 | 1.000 |
| 64 | Output only, half dropped | 6.971 | 6.991 | .267 | .995 |
| 96 | None | - | 6.956 | .484 | .981 |
| 96 | All 3 places | 6.677 | 7.602 | 0 | .495 |
| 96 | Output only | 6.777 | 6.988 | 0 | .995 |
| 96 | Output only, half dropped | 6.863 | 6.942 | .434 | .996 |

```bash
python3 mixed.py --hints 0 --width 96
python3 mixed.py --sites 0,1,2 --width 96
python3 mixed.py --sites 2 --width 96
python3 mixed.py --sites 2 --drop .5 --width 96
```

## How rare a fact must be (`rare.py`)

Accuracy without hints by how many times each of 200 facts per group was stated. Chance is 1%. With hints every group scores 100%. A model trained with hints scores at most 2.5% with hints off at every count up to 256. Runs took 74 to 311 s.

| Statements | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| Width 32 | .030 | .045 | .130 | .380 | .820 | .995 |
| Width 64 | .060 | .165 | .535 | .870 | .990 | 1.000 |
| Width 128 | .125 | .310 | .735 | .970 | 1.000 | 1.000 |
| Width 64, 4 times more training | - | - | .485 | .885 | .995 | 1.000 |

```bash
python3 rare.py --hints 0 --width 64
python3 rare.py --hints 1 --width 64
python3 rare.py --hints 0 --width 64 --repeat 4
```

## Where the web text gain sits (`web_split.py`)

Width 40, 1440 steps, 65K validation tokens, total gain 0.357. The two blocks split the same tokens.

| Group | Share of tokens | Share of gain | Gain per token |
|---|---|---|---|
| No match | 8% | -4% | -0.17 |
| Longest match 2 to 4 | 80% | 72% | 0.32 |
| Longest match 5 | 7% | 11% | 0.56 |
| Longest match 6 | 4% | 12% | 1.05 |
| Longest match 10 | 0.6% | 5% | 3.16 |
| Longest match 20 | 0.3% | 3% | 3.98 |
| Match seen once | 19% | 28% | 0.53 |
| Match seen 2 to 15 times | 35% | 39% | 0.40 |
| Match seen 16 or more times | 38% | 37% | 0.35 |

Hints that are right cover 18% of tokens and give 125% of the gain; wrong hints take back 20%. True next tokens seen 1K to 100K times hold 74% of the gain on 46% of tokens.

```bash
python3 web_run.py --policy none --steps 1440 --width 40 --hint-lr 0.1 --val-chunks 1024 --lr .02 --dump none
python3 web_run.py --policy page --size 7 --levels 2,3,4,5,6,10,20 --steps 1440 --width 40 --hint-lr 0.1 --val-chunks 1024 --lr .02 --dump all
python3 web_split.py
```

## Web text (`web_rows.py`, `web_run.py`)

FineWeb GPT-2 tokens, corpus 100M tokens, width 40, 4 layers, learning rate 0.02. Hints leave out the token's own page.

| Steps | No hints | Lengths 6, 10, 20 | Lengths 2 to 20 |
|---|---|---|---|
| 90 | 7.871 | 7.816 | 7.244 |
| 360 | 7.398 | 7.333 | 6.879 |
| 1440 | 6.916 | 6.872 | 6.580 |

The vocabulary grows with the steps, so compare within a row. With hints off, the model trained on lengths 2 to 20 scores 7.754 against 6.916 at 1440 steps.

```bash
mkdir data
curl -L -o data/fineweb_train_000001.bin https://huggingface.co/datasets/kjj0/fineweb100B-gpt2/resolve/main/fineweb_train_000001.bin
curl -L -o data/fineweb_val_000000.bin https://huggingface.co/datasets/kjj0/fineweb100B-gpt2/resolve/main/fineweb_val_000000.bin
for n in 2 3 4 5 6 10 20; do python3 web_rows.py $n; done
python3 web_run.py --policy page --size 7 --levels 2,3,4,5,6,10,20 --steps 1440 --width 40 --hint-lr 0.1 --lr .02
```

## Shakespeare (`shakespeare.py`)

Put Tiny Shakespeare at `data/shakespeare.txt`. Width 64, 200 steps, batch 8, best of three learning rates.

| Training hints | Hints on | Hints off |
|---|---|---|
| None | - | 2.165 |
| Own token kept | 3.525 | 4.735 |
| Earlier steps | 1.890 | 2.307 |
| Leave-one-out | 1.843 | 2.508 |
| **Leave-one-out, half dropped** | **1.805** | **2.256** |

All other tables are in the [paper](paper/main.pdf).
