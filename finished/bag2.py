"""Order-free lookup by whole words: how far does a word table get on reworded questions, with no model?

python bag2.py --pages 5000 --window 2 --title 1 --stem 1

Every unit (a few sentences, with its page title) is stored under each rare word it holds. A question adds up the weight of its
rare words over the units holding them. The candidates are the rare words of the best units not already in the question.
"""
import argparse, math, os, re, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--pages", type=int, default=5000)
p.add_argument("--rare", type=float, default=0.01, help="a word is rare if under this share of units holds it")
p.add_argument("--window", type=int, default=1, help="sentences in one unit")
p.add_argument("--title", type=int, default=0, help="1 = every unit also holds its page title")
p.add_argument("--stem", type=int, default=0, help="1 = cut common word endings")
p.add_argument("--top", type=int, default=5, help="best units that give candidates")
p.add_argument("--show", type=int, default=0)
p.add_argument("--model", type=int, default=0, help="1 = let the finished model pick among the candidates, and compare with exact hints and retrieval")
p.add_argument("--cands", type=int, default=10, help="candidates that get a higher score")
p.add_argument("--bumps", default="3,6,12", help="how much candidates' scores rise")
p.add_argument("--reason", type=int, default=100)
a = p.parse_args()
argv, sys.argv = sys.argv, ["hints", "none"]; sys.path.insert(0, HERE)
import hints as H
sys.argv = argv
WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.U)


def norm(w):
    w = w.lower()
    if w.endswith("'s"):
        w = w[:-2]
    if a.stem and len(w) > 4:
        for e in ("ing", "ed", "es", "s"):
            if w.endswith(e) and len(w) - len(e) >= 3:
                return w[:-len(e)]
    return w


t0 = time.time(); A, B, C = H.load_pages(); ALL = {r["title"]: r for r in A + B + C}; want = {x[0] for x in H.FACTS}
rows = [ALL[t] for t in want if t in ALL] + [r for r in A if r["title"] not in want][:a.pages]
text, unit = [], []
for r in rows:
    S = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", r["text"]) if len(s.strip()) > 10]
    for i in range(0, len(S), a.window):
        x = " ".join(S[i:i + a.window]); text.append(x); unit.append({norm(w) for w in WORD.findall((r["title"] + " " if a.title else "") + x)})
df = {}
for u in unit:
    for w in u:
        df[w] = df.get(w, 0) + 1
N = len(unit); CUT = a.rare * N; index = {}
for n, u in enumerate(unit):
    for w in u:
        if df[w] < CUT:
            index.setdefault(w, []).append(n)
print("units", N, "rare words", len(index), "entries", sum(map(len, index.values())), round(time.time() - t0, 1), "s", flush=True)
idf = lambda w: math.log(N / df[w])


def ask(q):
    Q = {norm(w) for w in WORD.findall(q)}; sc = {}
    for w in Q:
        for n in index.get(w, []):
            sc[n] = sc.get(n, 0) + idf(w)
    best = sorted(sc, key=lambda n: -sc[n])[:a.top]; cand = {}
    for n in best:
        for w in unit[n]:
            if w in index and w not in Q:
                cand[w] = cand.get(w, 0) + sc[n] * idf(w)
    return best, sorted(cand, key=lambda w: -cand[w]), Q


r = dict(unit1=0, unitk=0, top1=0, top10=0, top50=0); M = 0
for title, same, other, ans in H.FACTS:
    if title not in ALL:
        continue
    M += 1; w = norm(WORD.findall(ans)[0]); best, cand, Q = ask(other); holds = [ans.lower() in text[n].lower() for n in best]
    rank = next((i for i, c in enumerate(cand) if c.startswith(w)), None)
    r["unit1"] += bool(holds) and holds[0]; r["unitk"] += any(holds); r["top1"] += rank == 0; r["top10"] += rank is not None and rank < 10; r["top50"] += rank is not None and rank < 50
    if a.show:
        print(f"  {title[:34]:34} want {ans:9} rank {rank} of {len(cand)} unit {holds} top {cand[:5]}  missing {[x for x in Q if x not in df]}", flush=True)
print(f"pages {a.pages} window {a.window} title {a.title} stem {a.stem} top {a.top}: of {M} best unit holds answer {r['unit1']}, top units {r['unitk']}, candidate top1 {r['top1']} top10 {r['top10']} top50 {r['top50']}, {round(time.time() - t0, 1)} s", flush=True)


if a.model:
    import json
    import mlx.core as mx
    from mlx_lm import load
    t1 = time.time(); model, _ = load(os.path.join(HERE, "model")); tok = H.tokenizer(); print("model", round(time.time() - t1, 1), "s", flush=True)
    s_ = np.concatenate(H.encode(tok, rows)); tb = [H.table(s_, m) for m in H.LV]; w8 = np.load(os.path.join(HERE, "trust1.npy")); print("exact table", len(s_), round(time.time() - t1, 1), "s", flush=True)
    surface = {}
    for n, u in zip(range(N), text):
        for x in WORD.findall(u):
            surface.setdefault(norm(x), {}); surface[norm(x)][x] = surface[norm(x)].get(x, 0) + 1

    def tokens(words):
        T = set()
        for c in words:
            x = max(surface.get(c, {c: 1}).items(), key=lambda z: z[1])[0]
            T.add(tok.encode(" " + x, add_special_tokens=False).ids[0]); T.add(tok.encode(x, add_special_tokens=False).ids[0])
        return T

    def gen(q, n, bump=0, cands=()):
        s = np.array(tok.encode(q, add_special_tokens=False).ids, dtype=np.int64); got = []
        for _ in range(n):
            lo = np.array(model(mx.array(s[None])).astype(mx.float32)[0, -1])
            if bump:
                for t in cands(tok.decode(s.tolist())):
                    lo[t] += bump
            got.append(int(lo.argmax())); s = np.r_[s, got[-1]]
        return tok.decode(got).strip()

    def bag(q):
        return tokens(ask(q)[1][:a.cands])

    def ok(x, c):
        return x.strip().startswith(c.strip())

    F = [x for x in H.FACTS if x[0] in ALL]; rng = np.random.default_rng(2); out = {}
    arms = {"off": lambda q, c: ok(gen(q, 3), c), "exact": lambda q, c: ok(H.answer(model, tok, tb, w8, q, 3, 1), c),
            "rag1": lambda q, c: ok(gen(text[ask(q)[0][0]] + "\n\n" + q, 3), c), "rag3": lambda q, c: ok(gen("\n".join(text[n] for n in ask(q)[0][:3]) + "\n\n" + q, 3), c)}
    for b in a.bumps.split(","):
        arms["bag" + b] = (lambda b: lambda q, c: ok(gen(q, 3, float(b), bag), c))(b)
    for name, f in arms.items():
        t2 = time.time(); same = sum(f(x[1], x[3]) for x in F); other = sum(f(x[2], x[3]) for x in F); out[name] = (same, other)
        print(f"{name:7} same words {same} of {len(F)}, other words {other} of {len(F)}, {round(time.time() - t2, 1)} s", flush=True)
    C_ = H.comparison(a.reason, rng); S_ = H.sums(a.reason, rng)
    for name in ["off"] + ["bag" + b for b in a.bumps.split(",")] + ["rag1"]:
        t2 = time.time(); f = {"off": lambda q, n: gen(q, n)}.get(name) or ((lambda q, n: gen(text[ask(q)[0][0]] + "\n\n" + q, n) if ask(q)[0] else gen(q, n)) if name == "rag1" else (lambda b: lambda q, n: gen(q, n, b, bag))(float(name[3:])))
        c = sum(ok(f(q, 2), r) for q, r in C_); d = sum(ok(f(q, 4), r) for q, r in S_)
        print(f"{name:7} comparison {c} of {len(C_)}, sums {d} of {len(S_)}, {round(time.time() - t2, 1)} s", flush=True)
