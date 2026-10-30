# Results

Each setting was run once. Loss is cross-entropy on the next token; lower is better.

## Why hints block reasoning, and a switch that fixes it (`gate.py`, `mixed.py`)

Small test: width 32, 2 layers, 1000 steps, half facts and half comparison lines, "a > b . b > c . top ? a .", about 5 seconds, one run each. Hints set by hand at chosen words of comparison lines, in training and testing. Every row trains each step with and without hints. Accuracy is comparison with hints on.

| Hints on comparison lines | Comparison |
|---|---|
| On everywhere | .764 |
| On only at the answer word | .996 |
| Off only at the answer word | .516 |
| Off at ">" | .671 |
| Off at names | .798 |
| Off at dots | .968 |
| Off at the dot between the statements | 1.000 |
| Off on the whole line | 1.000 |

At the dot between the statements the hint says "top" and is right 0% of the time. Turning off only that hint gives .991, 1.000 and .974 at width 64, 4 layers and 4000 steps, but .553 at width 128, so it is not the only cause at every size. Keeping only hints with one sure next token gives .521. With hints always on and no second pass: .545, with the first name of the line picked 94% of the time.

Switch: a linear layer reads the hidden state from a pass without hints; the hint is multiplied by its sigmoid. Learned from the loss alone it stays at 1.0 and comparison is .769. Taught to predict whether the longest hint equals the next token, it drops to .03 at the harmful dot.

| Setting | Switch, input | Switch, all three | With and without hints | Hints at output |
|---|---|---|---|---|
| Width 16 | .546 | .503 | - | - |
| Width 32 | .994 | .636 | .764 | .995 |
| Width 64 | .996 | .964 | .547 | 1.000 |
| Width 128 | .959 | .955 | .639 | 1.000 |
| 4 layers | .913 | .664 | .512 | .999 |
| 4000 steps | .982 | .933 | .663 | 1.000 |

A switch that is fully on or off gives comparison .996 and facts .559. A switch that reads the hidden state where the hint enters, with no second pass, was .518 at width 128 (plain hints .611) and .701 at 4000 steps (.730).

One model on web text and the made-up language, switch at all three places, `python mixed.py --aux 1 --right 1 --width 64`:

| Width | Web, no hints | Web, plain hints | Web, switch | Web, switch off | Comparison, plain/switch | Common fact off | Held-out sum, no hints/switch |
|---|---|---|---|---|---|---|---|
| 32 | 7.132 | 6.824 | 6.781 | 7.082 | .496/.905 | .014 | .493/0 |
| 64 | 6.991 | 6.714 | **6.644** | 6.952 | .496/1.000 | .621 | .350/.251 |
| 96 | 6.956 | 6.677 | **6.582** | 6.918 | .495/1.000 | .957 | .484/.094 |

Training length at width 64, plain hints against switch: 360 steps 6.921 and 7.155, 1440 steps 6.816 and 6.819, 5760 steps 6.714 and 6.644. Comparison is not learned below 5760 steps. Switch with no training pass without hints, 1440 steps: 6.818, comparison .007; not run at 5760. Switch at the input only, width 64: web 6.671, comparison .495.

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

Accuracy without hints by how many times each of 200 facts per group was stated. Chance is 1%. With hints every group scores 100%. A model trained with hints scores at most 2.5% with hints off at every count up to 256. Runs took 42 to 311 s.

| Statements | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| Width 16 | .000 | .035 | .005 | .065 | .105 | .260 |
| Width 24 | .020 | .025 | .075 | .175 | .395 | .885 |
| Width 32 | .030 | .045 | .130 | .380 | .820 | .995 |
| Width 48 | .055 | .130 | .350 | .775 | 1.000 | 1.000 |
| Width 64 | .060 | .165 | .535 | .870 | .990 | 1.000 |
| Width 96 | .120 | .275 | .670 | .935 | 1.000 | 1.000 |
| Width 128 | .125 | .310 | .735 | .970 | 1.000 | 1.000 |
| Width 192 | .185 | .435 | .775 | .965 | .995 | 1.000 |
| Width 64, 2 layers | .070 | .175 | .470 | .850 | .995 | 1.000 |
| Width 64, 8 layers | .075 | .175 | .530 | .865 | 1.000 | 1.000 |
| Width 64, 100 facts per count | .090 | .240 | .480 | .860 | 1.000 | 1.000 |
| Width 64, 400 facts per count | .068 | .145 | .360 | .770 | .965 | 1.000 |
| Width 64, 4 times more training | - | - | .485 | .885 | .995 | 1.000 |

```bash
python3 rare.py --hints 0 --width 64
python3 rare.py --hints 1 --width 64
python3 rare.py --hints 0 --width 64 --repeat 4
```

## Keeping a model memorizing while it uses hints (`rare.py`, `mixed.py`)

Accuracy with hints off at width 64, facts only, by statements in training. With hints on every row scores at least 99%, except no hints in the last quarter, which falls to 67% at 256 statements.

| Training | 1 | 4 | 8 | 16 | 32 | 64 | 256 |
|---|---|---|---|---|---|---|---|
| No hints | .060 | .535 | .870 | .990 | 1.000 | 1.000 | 1.000 |
| Hints always | .000 | .000 | .000 | .000 | .000 | .005 | .005 |
| 25% of tokens without hints (`--drop .25`) | .005 | .010 | .015 | .005 | .010 | .140 | .700 |
| 50% | .020 | .010 | .030 | .060 | .285 | .915 | 1.000 |
| 75% | .045 | .235 | .555 | .960 | 1.000 | 1.000 | 1.000 |
| 90% | .070 | .425 | .785 | .985 | 1.000 | 1.000 | 1.000 |
| Share rises 0 to 100% (`--ramp 1`) | .010 | .065 | .195 | .505 | .870 | .995 | 1.000 |
| No hints in last half (`--off-at .5`) | .045 | .225 | .515 | .865 | .995 | 1.000 | 1.000 |
| No hints in last quarter (`--off-at .75`) | .005 | .040 | .055 | .175 | .375 | .765 | 1.000 |
| Hints at output only (`--sites 2`) | .020 | .000 | .010 | .000 | .010 | .030 | .010 |
| Output only, 50% without hints | .050 | .250 | .490 | .815 | 1.000 | 1.000 | 1.000 |
| **Every step with and without hints (`--aux 1`)** | .065 | .475 | .890 | .985 | 1.000 | 1.000 | 1.000 |
| Table keeps text seen under 16 times (`--max-seen 5`) | .015 | .025 | .000 | 1.000 | 1.000 | 1.000 | 1.000 |

The same fixes on web text and reasoning in one model, width 64, 5760 steps. Gain is the share of the 0.277 web gain of hints at all three places. Fact off is a common fact with hints off.

| Training | Web on | Web off | Sum on/off | Comparison on/off | Fact off | Gain | Seconds |
|---|---|---|---|---|---|---|---|
| No hints | - | 6.991 | .350 | .971 | .780 | - | 727 |
| All three places | 6.714 | 7.637 | .048/.000 | .496/.014 | .012 | 100% | 852 |
| **With and without hints (`--aux 1`)** | **6.726** | **6.952** | .040/.403 | .500/.960 | .070 | 96% | 1249 |
| Output only, with and without | 6.924 | 6.980 | .270/.270 | .979/.979 | .036 | 24% | 1251 |
| 75% of tokens without hints (`--drop-all .75`) | 6.868 | 6.991 | .222/.222 | .914/.774 | .270 | 44% | 877 |
| Table under 16 times (`--max-seen 5`) | 7.010 | 7.077 | .581/.581 | .495/.505 | 1.000 | 0% | 881 |
| Table under 64 times (`--max-seen 6`) | 6.933 | 7.132 | .349/.349 | .499/.488 | 1.000 | 21% | 880 |
| Under 16, with and without | 6.985 | 6.972 | .148/.148 | .543/.971 | 1.000 | 2% | 1267 |

```bash
python3 rare.py --aux 1 --width 64
python3 mixed.py --aux 1 --width 64
```

The closest work we found drops privileged inputs in reinforcement learning (arXiv:2005.09220). R-Drop runs two passes per batch on the same input (NeurIPS 2021). We found no test of these methods on next-token hints.

## Hints help text the model itself has not seen (`web_seen.py`)

The earlier split counts sightings in the 100M table corpus. The model trained on 737K tokens, 1/135 of that. This split counts sightings in the model's own training, for the 88.6% of tokens whose longest match fits inside their chunk.

| Times the model saw the context | Share of tokens | Share of gain | Gain per token |
|---|---|---|---|
| 0 | 67.2% | 79.2% | 0.421 |
| 1 | 7.9% | 8.3% | 0.377 |
| 2 to 3 | 4.7% | 5.1% | 0.393 |
| 4 to 15 | 5.0% | 3.7% | 0.265 |
| 16 to 63 | 2.3% | 0.9% | 0.138 |
| 64 or more | 1.5% | -0.1% | -0.026 |

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
