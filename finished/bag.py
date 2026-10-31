"""Order-free hints: a table keyed by pairs of rare tokens, on reworded questions.

python bag.py --pages 5000

Key = sorted pair of rare tokens in a sentence, value = the sentences holding that pair. Reading a question, every pair of its
rare tokens picks sentences, and rare tokens of the best sentences that are not already in the question become candidates.
"""
import argparse, os, re, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--pages", type=int, default=5000)
p.add_argument("--rare", type=float, default=0.01, help="a token is rare if under this share of sentences holds it")
p.add_argument("--keep", type=int, default=8, help="rarest tokens of a sentence or question used for keys")
p.add_argument("--size", type=int, default=2, help="tokens in a key")
p.add_argument("--window", type=int, default=1, help="sentences joined into one unit")
p.add_argument("--title", type=int, default=0, help="1 = every unit also holds its page title")
p.add_argument("--idf", type=float, default=0, help="candidate score is divided by this power of the token's sentence count")
p.add_argument("--show", type=int, default=1)
a = p.parse_args()
argv, sys.argv = sys.argv, ["hints", "none"]; sys.path.insert(0, HERE)
import hints as H
sys.argv = argv
from itertools import combinations

t0 = time.time(); tok = H.tokenizer(); A, B, C = H.load_pages(); print("pages", len(A), round(time.time() - t0, 1), flush=True)
ALL = {r["title"]: r for r in A + B + C}; want = {x[0] for x in H.FACTS}
rows = [ALL[t] for t in want if t in ALL] + [r for r in A if r["title"] not in want][:a.pages]
sent = []
for r in rows:
    S = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", r["title"] + "\n" + r["text"]) if len(s.strip()) > 10]
    sent += [(r["title"] + ". " if a.title else "") + " ".join(S[i:i + a.window]) for i in range(0, len(S), a.window)]
ids = [np.array(e.ids) for e in tok.encode_batch(sent, add_special_tokens=False)]
print("fact pages", len(want & set(ALL)), "sentences", len(sent), "tokens", sum(map(len, ids)), round(time.time() - t0, 1), flush=True)
V = 1 + max(int(x.max()) for x in ids if len(x)); df = np.zeros(V, dtype=np.int64)
for x in ids:
    df[np.unique(x)] += 1
CUT = a.rare * len(ids); word = lambda t: tok.decode([int(t)]).strip().lower()
good = np.array([df[i] < CUT and df[i] > 0 and len(word(i)) > 1 for i in range(V)])
print("rare tokens", int(good.sum()), "of", V, round(time.time() - t0, 1), flush=True)


def rarest(x):
    u = np.unique(x); u = u[(u < V)]; u = u[good[u]]
    return u[np.argsort(df[u])][:a.keep]


index = {}
for n, x in enumerate(ids):
    for k in combinations(sorted(rarest(x).tolist()), a.size):
        index.setdefault(k, []).append(n)
print("keys", len(index), "entries", sum(map(len, index.values())), round(time.time() - t0, 1), flush=True)


def candidates(q):
    """Rare tokens by score: sum over sentences of (shared keys)^2, each key weighted by how few sentences hold it."""
    sc = {}
    for k in combinations(sorted(rarest(q).tolist()), a.size):
        P = index.get(k, [])
        for n in P:
            sc[n] = sc.get(n, 0) + 1 / len(P) ** .5
    words = {}; seen = set(q.tolist())
    for n, v in sorted(sc.items(), key=lambda z: -z[1])[:20]:
        for t in np.unique(ids[n]):
            if t < V and good[t] and t not in seen:
                words[int(t)] = words.get(int(t), 0) + v / df[t] ** a.idf
    return sorted(words, key=lambda z: -words[z]), sc


tok_all, tb, *_ = H.build_tables(tok)
res = dict(top1=0, top10=0, top50=0, sent=0, exact1=0, cands=[]); n = 0
for title, same, other, ans in H.FACTS:
    if title not in ALL:
        continue
    n += 1; q = np.array(tok.encode(other, add_special_tokens=False).ids, dtype=np.int64); w = ans.lower()
    C_, sc = candidates(q); hit = lambda t: word(t) and w.startswith(word(t)) and len(word(t)) >= min(2, len(w))
    rank = next((i for i, t in enumerate(C_) if hit(t)), None)
    ht, _ = H.lookup(tb, q); e = [word(t) for t in ht[:, -1] if t >= 0]; ex = bool(e) and w.startswith(e[-1]) and len(e[-1]) > 0
    best = max(sc, key=sc.get) if sc else None
    res["top1"] += rank == 0; res["top10"] += rank is not None and rank < 10; res["top50"] += rank is not None and rank < 50; res["exact1"] += ex
    res["sent"] += best is not None and ans.lower() in sent[best].lower(); res["cands"].append(len(C_))
    if a.show:
        print(f"  {title[:34]:34} rank {rank} of {len(C_)}  exact {e[-1:] }  want {ans}  top {[word(t) for t in C_[:5]]}", flush=True)
print(f"pages {a.pages} rare {a.rare} keep {a.keep} size {a.size} window {a.window} title {a.title} idf {a.idf}: of {n} exact top {res['exact1']}, bag top1 {res['top1']}, top10 {res['top10']}, top50 {res['top50']}, best sentence holds answer {res['sent']}, median candidates {int(np.median(res['cands']))}, {round(time.time() - t0, 1)} s", flush=True)
