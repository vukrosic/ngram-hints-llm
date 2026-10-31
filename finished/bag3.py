"""Small memory test: does the finished model answer reworded questions with word-set candidates, compared with retrieval?

python bag3.py --pages 1000

The table holds only a thousand pages plus the new pages, so this stays small. Candidates get a higher score at the first answer token.
"""
import argparse, json, math, os, re, sys, time
import numpy as np
import pyarrow as pa

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--pages", type=int, default=1000)
p.add_argument("--bump", type=float, default=6)
p.add_argument("--cands", type=int, default=10)
p.add_argument("--reason", type=int, default=20)
p.add_argument("--near", type=int, default=0, help="1 = no model, rank candidates by how close they sit to the question's rare words")
p.add_argument("--snip", default="", help="snippet lengths in words, placed before the question, best window of the best units")
p.add_argument("--why", type=int, default=0, help="1 = one model pass per question, show why the answer was or was not chosen")
a = p.parse_args()
argv, sys.argv = sys.argv, ["hints", "none"]; sys.path.insert(0, HERE)
import hints as H
sys.argv = argv
WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.U)


def norm(w):
    w = w.lower()
    if w.endswith("'s"):
        w = w[:-2]
    if len(w) > 4:
        for e in ("ing", "ed", "es", "s"):
            if w.endswith(e) and len(w) - len(e) >= 3:
                return w[:-len(e)]
    return w


t0 = time.time(); rows = [json.loads(l) for l in open(os.path.join(HERE, "pages.jsonl"))]; want = {x[0] for x in H.FACTS}; have = {r["title"] for r in rows}
r_ = pa.ipc.open_stream(open(os.path.join(HERE, "wiki", "part.arrow"), "rb")); n_ = 0
while n_ < a.pages:
    b = r_.read_next_batch()
    for t, x in zip(b["title"].to_pylist(), b["text"].to_pylist()):
        if (t in want and t not in have) or n_ < a.pages:
            rows.append(dict(title=t, text=x)); n_ += 1
del r_
text, unit = [], []
for r in rows:
    S = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", r["text"]) if len(s.strip()) > 10]
    for i in range(0, len(S), 3):
        x = " ".join(S[i:i + 3]); text.append(x); unit.append({norm(w) for w in WORD.findall(r["title"] + " " + x)})
titles = {r["title"] for r in rows}; del rows
df = {}
for u in unit:
    for w in u:
        df[w] = df.get(w, 0) + 1
N = len(unit); index = {}
for n, u in enumerate(unit):
    for w in u:
        if df[w] < .01 * N:
            index.setdefault(w, []).append(n)
idf = lambda w: math.log(N / df[w])
print("units", N, "fact pages", len(want & titles), round(time.time() - t0, 1), "s", flush=True)


def ask(q):
    Q = {norm(w) for w in WORD.findall(q)}; sc = {}
    for w in Q:
        for n in index.get(w, []):
            sc[n] = sc.get(n, 0) + idf(w)
    best = sorted(sc, key=lambda n: -sc[n])[:5]; cand = {}; form = {}
    for n in best:
        for x in WORD.findall(text[n]):
            form.setdefault(norm(x), x)
        for w in unit[n]:
            if w in index and w not in Q:
                cand[w] = cand.get(w, 0) + sc[n] * idf(w)
    return best, [form.get(w, w) for w in sorted(cand, key=lambda w: -cand[w])]


if a.near:
    F = [x for x in H.FACTS if x[0] in titles]; R = {"list": [0, 0, 0], "near": [0, 0, 0]}
    for title, same, q, ans in F:
        best, base = ask(q); Q = {norm(w) for w in WORD.findall(q)}; w_ = norm(WORD.findall(ans)[0]); sc = {}
        for r_n, n in enumerate(best):
            W = [norm(x) for x in WORD.findall(text[n])]; P = [(i, x) for i, x in enumerate(W) if x in Q and x in index]
            for i, x in enumerate(W):
                if x in index and x not in Q:
                    sc[x] = sc.get(x, 0) + sum(idf(y) / (1 + abs(i - j)) for j, y in P)
        near = sorted(sc, key=lambda c: -sc[c]); a_ = [norm(c) for c in base]
        for name, L in (("list", a_), ("near", near)):
            r = next((i for i, c in enumerate(L) if c.startswith(w_)), None)
            for k, top in enumerate((1, 5, 10)):
                R[name][k] += r is not None and r < top
    print("right word in top 1, 5, 10 of", len(F), R, round(time.time() - t0, 1), "s", flush=True); raise SystemExit
import mlx.core as mx
from mlx_lm import load
mx.set_cache_limit(256 * 1024 * 1024)
model, _ = load(os.path.join(HERE, "model")); tok = H.tokenizer(); print("model", round(time.time() - t0, 1), "s", flush=True)


def gen(q, n, bump=None):
    s = np.array(tok.encode(q, add_special_tokens=False).ids, dtype=np.int64); got = []
    for i in range(n):
        lo = np.array(model(mx.array(s[None])).astype(mx.float32)[0, -1])
        if bump is not None and i == 0:
            for t in bump:
                lo[t] += a.bump
        got.append(int(lo.argmax())); s = np.r_[s, got[-1]]
    return tok.decode(got).strip()


def cands(q):
    T = set()
    for c in ask(q)[1][:a.cands]:
        T.add(tok.encode(" " + c, add_special_tokens=False).ids[0]); T.add(tok.encode(c, add_special_tokens=False).ids[0])
    return T


if a.why:
    F = [x for x in H.FACTS if x[0] in titles]; T_ = dict(missing=0, wins_alone=0, loses_to_model=0, loses_to_candidate=0); pick = {10: 0, 50: 0, 1000: 0}
    for title, same, q, ans in F:
        s = np.array(tok.encode(q, add_special_tokens=False).ids, dtype=np.int64); lo = np.array(model(mx.array(s[None])).astype(mx.float32)[0, -1])
        C_ = ask(q)[1]; want = {tok.encode(" " + ans, add_special_tokens=False).ids[0], tok.encode(ans, add_special_tokens=False).ids[0]}; ids_ = []
        for c in C_:
            ids_.append((c, tok.encode(" " + c, add_special_tokens=False).ids[0])); ids_.append((c, tok.encode(c, add_special_tokens=False).ids[0]))
        rank = next((i // 2 for i, (c, t) in enumerate(ids_) if t in want), None)
        for k in pick:
            top = max((t for c, t in ids_[:2 * k]), key=lambda t: lo[t], default=None); pick[k] += top in want
        ten = [t for c, t in ids_[:20]]; best = max(ten, key=lambda t: lo[t], default=-1); w0 = max(want, key=lambda t: lo[t]); gap = float(lo.max() - lo[w0])
        kind = "missing" if rank is None or rank >= 10 else "wins_alone" if best in want and gap < a.bump else "loses_to_candidate" if best not in want else "loses_to_model"
        T_[kind] += 1
        print(f"  {title[:30]:30} want {ans:9} candidate rank {rank}  model alone {tok.decode([int(lo.argmax())])!r}  gap {gap:.1f}  best of 10 {tok.decode([int(best)])!r}  {kind}", flush=True)
    print(T_, "best-scored word when only candidates may be chosen, right of", len(F), pick, round(time.time() - t0, 1), "s", flush=True); raise SystemExit
def snippet(q, n):
    """The n words of the best units holding the most question weight."""
    Q = {norm(w) for w in WORD.findall(q)}; top = (0, "")
    for u in ask(q)[0]:
        W = text[u].split(); v = [idf(norm(x)) if norm(x) in Q and norm(x) in index else 0 for x in (re.sub(r"\W+$", "", y) for y in W)]
        for i in range(max(1, len(W) - n + 1)):
            c = sum(v[i:i + n])
            if c > top[0]:
                top = (c, " ".join(W[i:i + n]))
    return top[1]


if a.snip:
    F = [x for x in H.FACTS if x[0] in titles]
    for n in [int(c) for c in a.snip.split(",")]:
        P = [snippet(x[2], n) + "\n\n" + x[2] for x in F]; t1 = time.time(); R = [gen(q, 3) for q in P]; extra = np.mean([len(tok.encode(p[:-len(x[2])], add_special_tokens=False).ids) for p, x in zip(P, F)])
        print(f"snippet {n} words: other words {sum(r.strip().startswith(x[3]) for r, x in zip(R, F))} of {len(F)}, extra prompt tokens {extra:.0f}, {round(time.time() - t1, 1)} s", [r[:12] for r in R], flush=True)
    b = [text[ask(x[2])[0][0]] + "\n\n" + x[2] for x in F]; print("retrieval extra prompt tokens", round(float(np.mean([len(tok.encode(p[:-len(x[2])], add_special_tokens=False).ids) for p, x in zip(b, F)]))), flush=True); raise SystemExit
rag = lambda q: text[ask(q)[0][0]] + "\n\n" + q if ask(q)[0] else q
ok = lambda x, c: x.strip().startswith(c.strip())
F = [x for x in H.FACTS if x[0] in titles]
for name, f in (("off", lambda q: gen(q, 3)), ("candidates", lambda q: gen(q, 3, cands(q))), ("retrieval", lambda q: gen(rag(q), 3))):
    t1 = time.time(); R = [f(x[2]) for x in F]; print(f"{name:11} other words {sum(ok(r, x[3]) for r, x in zip(R, F))} of {len(F)}, {round(time.time() - t1, 1)} s", flush=True)
    if name != "off":
        print("  ", [r[:12] for r in R], flush=True)
Q = H.comparison(a.reason, np.random.default_rng(2))
print("comparison off", sum(ok(gen(q, 2), c) for q, c in Q), "candidates", sum(ok(gen(q, 2, cands(q)), c) for q, c in Q), "of", len(Q), round(time.time() - t0, 1), "s", flush=True)
