# Results

## SetGram: made-up biographies in web text (`bio_rows.py`, `bio_run.py`)

**Data.** 400 made-up people with syllable names. Each has a birth city (30 choices), a university (17) and a company (28), in four wordings:

| Wording | Example | Where it appears |
|---|---|---|
| 0 | " N was born in A." | the table, for all 400 people |
| 1 | " The city where N was born is A." | training text, people 0 to 199 |
| 2 | " Where was N born? A." | training text, people 0 to 199 (not with `--once`) |
| 3 | " Which city was N born in? A." | test only, never stored or trained |

With `--once`, each trained person gets one training wording, so their facts are in the table twice, like a rare web fact. The table holds the first 25M FineWeb training tokens, the packed training biographies and wording 0. Training rows are 4000 FineWeb rows of 64 tokens, with 2 biography rows in every batch of 8.

**Keys.** Exact keys are the last 2, 3, 4 or 6 tokens. SetGram keys are the sorted set of the last 2, 3 or 4 rare tokens (outside the 500 most frequent) within the 16 tokens before, plus a flag for whether the token just before is rare. A training token's own count is left out of its hint.

**Recall.** The right answer must be the top pick among all answers of that kind (chance about 4.5%), with the question as the prompt.

**Model.** Width 32, 2 layers, 4 heads, RoPE, QK-norm, SwiGLU, RMSNorm, no biases. Muon for hidden matrices, Adam for embeddings and memory tables. 120 steps, one run per setting.

**Baselines in `bio_run.py`** (simplified re-implementations, not the original code):

| Arm | What it is |
|---|---|
| `none` | no memory |
| `exact` | exact n-gram hints of the NanoGPT record #367, keys 2, 3, 4, 6 |
| `bag` | SetGram hints, keys 2, 3, 4 |
| `both` | exact and SetGram hints together |
| `engram` | Engram-style (arXiv 2601.07372): hashed 2- and 3-token suffixes, 50,021 buckets each, sigmoid gate, added after the first block |
| `target` | CoCoNTs-style (arXiv 2407.02819): the longest exact hint joins the training target with weight equal to its share; the model never sees hints |
| `memory` | memory layer with product keys (arXiv 1907.05242, 2412.09764): 128 × 128 keys, 16,384 values, top 32, added after the first block |
| `none` + test mix | infini-gram style (arXiv 2401.17377): at test, p = (1 − λs) p_model + λs on the hint token, s = the hint's share, λ chosen on web validation from .1, .3, .5, .9 |

**The table alone, no model.** Share of test questions whose top hint at the answer is right:

| Keys | Stored wording | New wording |
|---|---|---|
| Exact, 2 to 6 tokens | up to 98% (6 tokens) | 0% |
| SetGram, 3 rare tokens | | 67% to 79% |
| SetGram, 4 rare tokens | | 96% |

**Facts in the data twice** (`--once`). Web loss is with the memory on.

| Arm | Web loss | Trained people, new wording | Table-only people, new wording | Table-only people, stored wording |
|---|---|---|---|---|
| none | 7.807 | .067 | .038 | .032 |
| exact hints | 7.428 | .057 | .030 | .487 |
| Engram-style | 7.808 | .047 | .040 | .038 |
| CoCoNTs-style target | 7.761 | .060 | .043 | .047 |
| memory layer | 7.806 | .062 | .055 | .045 |
| none + exact mix at test | 7.348 | .067 | .038 | .980 |
| **SetGram hints** | 7.886 | **.812** | **.795** | **.757** |
| exact + SetGram hints | 7.458 | .478 | .702 | .845 |
| **none + SetGram mix at test** | 7.670 | **.963** | **.958** | **.958** |

**Facts in the data three times** (default rows):

| Arm | Web loss | Trained people, new wording | Table-only people, new wording | Table-only people, stored wording |
|---|---|---|---|---|
| none | 7.897 | .047 | .045 | .045 |
| exact hints | 7.483 | .050 | .033 | .395 |
| Engram-style | 7.899 | .052 | .048 | .038 |
| CoCoNTs-style target | 7.844 | .060 | .052 | .042 |
| none + exact mix at test | 7.397 | .047 | .045 | .980 |
| **SetGram hints** | 7.944 | **.923** | .050 | .057 |
| exact + SetGram hints | 7.500 | .898 | .005 | .287 |
| **none + SetGram mix at test** | 7.754 | **.963** | **.958** | **.958** |

Here table-only people fail with SetGram hints in training. Their hint has count 1, and on web text a count-1 hint is almost always wrong, so the model learns not to trust that kind. With facts in the data twice, trained facts also have count 1 hints that are right, and the model learns to trust them.

**Scaling** (SetGram hints, `--once`, table-only people, new wording):

| Steps (width 32) | 30 | 60 | 120 |
|---|---|---|---|
| Recall | .50 | .67 | .80 |

| Width (60 steps) | 32 (1.2M weights) | 64 | 128 (5.0M weights) |
|---|---|---|---|
| SetGram hints | .67 | .77 | .79 |
| exact hints | | | .043 |

Width covers 4×, not 8×. 240 steps and more were not run (over the 10 s per run limit).

```bash
python3 bio_rows.py            # data/bio_rows.npz, 27 s, 3 GB peak
python3 bio_rows.py --once     # data/bio_rows_once.npz
for arm in none exact bag both engram target memory; do python3 bio_run.py --arm $arm --steps 120 --rows bio_rows_once; done
for arm in none exact bag both engram target; do python3 bio_run.py --arm $arm --steps 120; done
python3 bio_run.py --arm bag --steps 30 --rows bio_rows_once
python3 bio_run.py --arm bag --steps 60 --width 128 --rows bio_rows_once
```

The `none` arm also prints the test-time mixes (`mix_exact`, `mix_bag`). It needs the two FineWeb files from the web text section below.

## SetGram on the made-up fact language (`facts.py --bag 1`)

Rare facts stored in the table in one wording, asked in another, never trained on. Content words are names, digits and relations. Right of 1000, 2 layers, hints at the input, middle and output:

| Hints | Width 32, 50 steps | 200 steps | 800 steps | Width 64, 200 steps | Width 128, 200 steps |
|---|---|---|---|---|---|
| none | .000 | .009 | .011 | .009 | .015 |
| exact | .009 | .007 | .005 | .010 | .008 |
| **SetGram** | **.737** | **.840** | **.850** | **.852** | **.853** |
| exact + SetGram | .684 | .762 | .778 | .649 | .634 |

The same facts in the stored wording: exact 1.0, SetGram .76 to .86. The SetGram table itself is right on 88%; the rest are collisions, where "x y z" and "z y x" share a key.

```bash
python3 facts.py --hints 0 --bag 1 --use 0,1,2,3 --width 32 --layers 2 --steps 200   # none
python3 facts.py --hints 1 --bag 1 --use 0,1,2,3 --width 32 --layers 2 --steps 200   # exact
python3 facts.py --hints 1 --bag 1 --use 4,5,6 --width 32 --layers 2 --steps 200     # SetGram
python3 facts.py --hints 1 --bag 1 --use 0,1,2,3,4,5,6 --width 32 --layers 2 --steps 200   # both
```

## SetGram keys on web text loss (`web_bag_rows.py`, `web_run.py`)

This version keeps the previous token exact and adds the set of the last 2 or 3 rare tokens in the 16 before. 25M-token table, hints at the input, middle and output. Validation loss with hints on:

| Steps, width | none | exact 6,10,20 (record) | + SetGram 2,3 | gain | exact 2 to 20 | + SetGram 2,3 | gain |
|---|---|---|---|---|---|---|---|
| 35, 32 | 8.532 | 8.523 | 8.431 | .092 | 8.121 | 8.115 | .007 |
| 70, 32 | 7.925 | 7.909 | 7.840 | .069 | 7.524 | 7.513 | .010 |
| 140, 32 | 7.755 | 7.743 | 7.697 | .047 | 7.355 | 7.348 | .006 |
| 70, 64 | 7.841 | 7.822 | 7.783 | .040 | 7.435 | 7.432 | .003 |
| 70, 96 | 7.778 | 7.765 | 7.734 | .031 | 7.384 | 7.375 | .009 |

On general web text, SetGram keys mostly repeat what short exact keys already give. Their use is facts asked in other words.

```bash
python3 web_bag_rows.py 2; python3 web_bag_rows.py 3
python3 web_run.py --policy loo --size 5 --levels 6,10,20,bag2,bag3 --width 32 --layers 2 --steps 70 --warmup 5
```

## CoCoNTs-style table targets on web text (`web_run.py --target`)

The longest 2-, 3- or 4-token exact match gives its top next token. The training target becomes (1 − w) the true token + w the table token, with w = T × the table's share. The model never sees hints. Validation loss, no table:

| Steps (width 32) | plain | T .25 | T .5 | T 1 |
|---|---|---|---|---|
| 35 | 8.636 | 8.597 | 8.579 | 8.571 |
| 70 | 7.945 | 7.898 | 7.882 | 7.883 |
| 140 | 7.764 | 7.712 | 7.693 | 7.696 |

At 70 steps the gain at T .5 is .063, .069 and .063 at widths 32, 64 and 96.

```bash
python3 web_run.py --policy loo --target 0.5 --levels 2,3,4 --width 32 --layers 2 --steps 70 --warmup 5
```

## Word-set lookups on a finished model (`finished/bag.py`, `bag2.py`, `bag3.py`)

MiniCPM5-2B, unchanged, with a table of 1000 Wikipedia pages plus 18 new made-up pages. 18 questions asked in new words. Whole words are the keys, page titles are added, and the best sentences give candidate answers.

| What the model gets | Right of 18 | Extra prompt tokens |
|---|---|---|
| Nothing | 1 | 0 |
| Candidate answers get a higher score | 4 | 0 |
| 10-word snippet from the best match | 3 | 15 |
| 20-word snippet | 7 | 28 |
| 40-word snippet | 8 | 54 |
| Retrieval, best passage in the prompt | 7 | 81 |

A finished model that never trained with the table cannot tell which candidate answers the question. A 20-word snippet matched retrieval with a third of the extra prompt.

```bash
python3 finished/bag3.py --pages 1000
python3 finished/bag3.py --pages 1000 --snip 10,20,40
```

Each setting was run once. Loss is cross-entropy on the next token; lower is better.

## How hints block reasoning (`gate.py`)

Hints on from step 600, `python gate.py --arm always --steps 1600 --late 600 --freeze 200 --every 400 --width 32`. Comparison on held-out names, hints on unless stated, one run each:

| Stage | Width 32 | Width 64 | Width 128 | 4 layers |
|---|---|---|---|---|
| 600 steps with no hints, hints off | .946 | .974 | .996 | .931 |
| Then 200 steps training only trust | .958 | .853 | .809 | .950 |
| Then 800 steps with hints on | .504 | .852 | .872 | .580 |
| Hints on from the start | .505 | .691 | .522 | .499 |

Facts after trust training are .80 with hints on and about .06 with hints off. With the shared table, after 600 steps the model scores .999 on training lines and .52 on held-out names.

Small table in training, full table at testing, `python gate.py --arm always --small 0.1 --width 32`. The training table reads the given share of lines the model does not train on. Test counts are scaled by that share.

| Share | Times the lines trained on | Width 32 | Width 64 | Width 128 |
|---|---|---|---|---|
| 2% | 0.1 | .781 | - | .957 |
| 5% | 0.25 | .996 | .996 | .994 |
| 10% | 0.5 | 1.000 | - | 1.000 |
| 20% | 1 | 1.000 | 1.000 | 1.000 |
| 30% | 1.6 | 1.000 | 1.000 | 1.000 |
| 40% | 2.1 | .997 | .990 | .990 |
| 50% | 2.6 | .999 | - | 1.000 |
| 60% | 3.2 | .972 | .984 | 1.000 |
| 70% | 3.7 | .897 | .945 | .785 |
| 75% | 3.9 | .549 | - | 1.000 |
| 80% | 4.2 | 1.000 | .877 | 1.000 |
| 90% | 4.7 | .986 | .976 | .823 |
| 95% | 5.0 | .982 | .504 | .541 |
| 100% | 5.3 | .523 | .597 | .503 |

Facts are .87 to .89 from 5% up. From 20% up the answer hint on training lines is right at the same rate at every size, .50 for the longest hint and .55 for any length.

At 100%, comparison is .502 at 4 layers. Training lines score near 1.0 in good and bad runs, so the break is not explained.

Separate table, `python gate.py --arm always --split 1 --width 32`: the table reads one half of the name triples and training uses the other half. Comparison is 1.000 at width 32, 64, 128 and 4 layers, and at 4000 steps at width 32. Facts .80 to .82. Comparison with hints off is .029, .067, 0 and .231. Each triple on one line (`--unique 1`): .996, .993, .997, .994, facts .80.

### The limit follows copies in the table

`python gate.py --arm always --small 1.0 --names 30 --width 32`. A copy is one more table line with the same names and the same statement order, so each triple has two versions. The name rows hold the same table size.

| Setting | Copies of each line | Width 32 | Width 128 |
|---|---|---|---|
| 14 names, full table | 35 | .552 | .560 |
| 20 names, 500 steps | 7.9 | .552 | .884 |
| 20 names, full table | 7.2 | .523 | .503 |
| 20 names, 90% of table | 6.5 | .986 | .823 |
| 20 names, 2000 steps | 5.8 | .920 | .720 |
| 20 names, 60% of table | 4.3 | .972 | 1.000 |
| 30 names, full table | 1.8 | .999 | 1.000 |
| 40 names, full table | 0.7 | 1.000 | 1.000 |

Explanations that failed: without length 6 hints the full table still blocks (`--levels 0,1,2`: .504, .508, .448 at widths 32, 64, 128), length 4 alone blocks (.492, .501), and one trust level for every hint (`--nokind 1`) gives .590, .529 at 100% and .982, .978 at 20%, facts .89. A 20% table in training and testing (`--testsmall 1`) keeps comparison at 1.000 with facts .662 and .750. Removing the hint at the word "top" from a blocked model (`--probe 1`) raises held-out comparison from .52 to .80 at width 32 and .50 to .61 at width 128.

### Copies of the training sentences block the skill at every name count

`python gate.py --arm always --small 1 --names 30 --chains 716000 --countscale 1 --width 32`. Lines made = training lines + 2 x (triples - 1000) x copies. Cells are width 32 / width 128.

| Names | 1 copies | 2 copies | 4 copies | 5 copies | 7 copies | 10 copies | 15 copies | 30 copies |
|---|---|---|---|---|---|---|---|---|
| 14 | .95/.95 | .99/.99 | 1.0/1.0 | .99/.99 | .96/.98 | .56/.53 | .98/.50 | .54/.57 |
| 20 | 1.0/1.0 | 1.0/1.0 | 1.0/1.0 | 1.0/1.0 | .81/.98 | .51/.71 | .69/.52 | .65/.48 |
| 30 | 1.0/1.0 | 1.0/1.0 | 1.0/.99 | .99/.99 | .99/.93 | 1.0/.54 | .51/.49 | .48/.51 |
| 40 | 1.0/1.0 | 1.0/1.0 | 1.0/1.0 | 1.0/1.0 | .99/1.0 | .84/.49 | .48/.48 | -- |

Without `--countscale 1`, comparison at 14 names falls to .51 to .72 at 1 to 7 copies, and facts are .73 at 14 names and .89 at 20 names. With it, facts at few copies are about .1 to .2 at those name counts.

| Test (20 names, 30 copies unless stated) | Width 32 | Width 128 |
|---|---|---|
| Default | .654 | .483 |
| Table reads other triples, 17 copies (`--split 1`) | 1.000 | 1.000 |
| Table reads other triples, 60 copies | 1.000 | 1.000 |
| 40 names, 2 copies, 1000 lines trained 16 times (`--pool 1000`) | .999 | 1.000 |
| Test lines added to the test table 30 times (`--testseen 30`) | 1.000 | 1.000 |
| Test lines added 7 times | .671 | .483 |
| No hint at the first dot (`--cut 3`) | .953 | 1.000 |
| No hint at the second dot (`--cut 7`) | .915 | .483 |
| Only the answer hint kept | .996 | .999 |
| No answer hint (`--cut 9`) | .483 | .648 |

A blocked model on held-out lines scores about 1.00 when the answer is the first word and about .00 otherwise. `--cut 3` at 14, 30 and 40 names: .917/.531, .946/.993, .908/1.000. Removing the hint at words 0, 1, 2, 4, 5, 6 or 8 left it blocked at both widths or one.

At 30 names, 2 copies gives 1.000 at widths 32, 64, 128, 256, at 4 layers and at 4000 steps. At 30 copies: .483, .491, .510, .483 at those widths, .995 at 4 layers width 64, and at 4000 steps .973 at width 32 and .502 at width 128.

## Hints set by hand, and a switch (`gate.py`, `mixed.py`)

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

## Hints on a finished model (`finished/`)

MiniCPM5-2B, 2.5 billion parameters, 4-bit MLX, weights unchanged. Hints only raise the output score of the hinted token. Hint lengths 2, 3, 4, 6. Trust has 840 numbers: length × count bucket × share bucket × how likely the model thinks the token is. Table, trust and test text share no file or page. One run per setting. The model file, Wikipedia part and tables are not in this repo; download `openbmb/MiniCPM5-2B-MLX` to `finished/model/` and a shard of `NeuML/wikipedia-20260401` to `finished/wiki/part.arrow`, and run `fetch.py` for the 2026 pages.

Commands: `python hints.py features`, `python hints.py trust`, `python hints.py reason`, `python domains.py`, `python more1.py`, `python more2.py`, `python more3.py`.

Wikipedia, 27.6M table tokens, 18,840 test places (table in the paper):

| Test | Off | On |
|---|---|---|
| Unseen text, next word right | 41.7% | 43.1% |
| Unseen text, loss | 3.145 | 3.086 |
| 2026 pages in the table, next word right | 41.6% | 83.0% |
| 2026 pages in the table, loss | 3.192 | 0.920 |
| 11 new facts, page's words | 1 | 6 |
| Same facts reworded | 1 | 1 |
| 100 comparison questions | 54 | 54 (0 changed) |
| 100 two-digit sums | 1 | 1 (25 changed) |

By kind of text, model reading 128 tokens, about 8,000 test places, new table and trust each row:

| Text | Table tokens | Loss off | Loss on | Top off | Top on | 6-token hint |
|---|---|---|---|---|---|---|
| transformers, unseen model folders | 6.3M | 1.608 | 0.839 | 66.2% | 83.9% | 81% |
| transformers | 5.3M | 1.697 | 1.067 | 65.7% | 79.2% | 69% |
| pandas | 3.8M | 1.596 | 1.525 | 67.0% | 68.9% | 38% |
| sympy | 6.0M | 1.708 | 1.586 | 64.4% | 67.9% | 35% |
| sklearn | 2.4M | 1.661 | 1.550 | 64.8% | 68.0% | 34% |
| networkx | 1.3M | 1.596 | 1.478 | 65.2% | 68.7% | 33% |
| scipy | 4.3M | 1.856 | 1.811 | 61.3% | 63.0% | 24% |
| Own Markdown notes, exact copies removed | 5.6M | 4.269 | 3.648 | 32.1% | 43.1% | 20% |
| sympy, 10% of table | 0.45M | 1.708 | 1.725 | 64.4% | 65.1% | 13% |
| Wikipedia, 10% of table | 2.6M | 3.126 | 3.133 | 42.0% | 42.6% | 4% |
| Table from transformers, test on torch | 7.3M | 2.093 | 2.136 | 58.6% | 58.4% | 4% |
| sympy, 1% of table | 0.04M | 1.708 | 1.755 | 64.4% | 64.4% | 3% |
| Wikipedia, 1% of table | 0.26M | 3.126 | 3.144 | 42.0% | 42.1% | 1% |

Model reading 512 tokens, scored on the last 120 places of each block:

| Text | Loss off | Loss on | Top off | Top on |
|---|---|---|---|---|
| sympy | 0.904 | 0.907 | 79.1% | 79.9% |
| transformers, unseen model folders | 0.993 | 0.646 | 77.1% | 85.7% |

5 tokens in a row, 96 prompts, 64 tokens of context: transformers unseen folders 25 to 40, sympy 26 to 25. Lookup 0.66 ms per token in Python, model 5.6 ms per token at 128 tokens at once, table of 6M tokens 234 MB.

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
