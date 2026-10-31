"""Add n-gram hints to a finished model without changing its weights.

python hints.py features         read text with the model and save what the trust step needs
python hints.py trust            learn the trust numbers and score web text with hints on and off
python hints.py reason           answer facts, comparison and sums with hints on and off

The table reads Wikipedia pages and stores the most common next token after each word group.
At the output, the score of the hinted token rises by a learned trust number that depends on the
hint length, how often the group was seen, how often that next token followed it, and how likely
the model already thinks that token is. Nothing inside the model changes.
"""
import argparse, json, os, re, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("mode")
p.add_argument("--trust-pos", type=int, default=40000)
p.add_argument("--test-pos", type=int, default=20000)
p.add_argument("--fact-pos", type=int, default=20000)
p.add_argument("--block", type=int, default=128)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--steps", type=int, default=400)
p.add_argument("--reason", type=int, default=100)
a = p.parse_args()
LV = [2, 3, 4, 6]; C_EDGES = np.array([1, 2, 4, 8, 16, 64]); P_EDGES = np.array([0.25, 0.5, 0.75, 0.9, 0.9999])
M_EDGES = np.array([0.01, 0.1, 0.5, 0.9]); NK = 42; NM = len(M_EDGES) + 1


def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(HERE, "model", "tokenizer.json"))


def wiki_pages():
    """The part of the April 2026 Wikipedia copy that was downloaded. The last batch is cut, so stop there."""
    import pyarrow as pa
    R = []; r = pa.ipc.open_stream(open(os.path.join(HERE, "wiki", "part.arrow"), "rb"))
    try:
        while True:
            b = r.read_next_batch(); R += [dict(title=t, text=x) for t, x in zip(b["title"].to_pylist(), b["text"].to_pylist())]
    except OSError:
        return R


def load_pages():
    """Table, trust and test pages share no page. Pages about 2026 events carry a root, and those in the table are the new facts."""
    R = [json.loads(l) for l in open(os.path.join(HERE, "pages.jsonl"))] + wiki_pages()
    np.random.default_rng(0).shuffle(R)
    n = len(R); return R[:int(n * .7)], R[int(n * .7):int(n * .85)], R[int(n * .85):]


def encode(tok, pages):
    return [np.array(e.ids, dtype=np.int64) for e in tok.encode_batch([r["title"] + "\n" + r["text"] for r in pages], add_special_tokens=False)]


def keys(s, m):
    """Hash of the m tokens that end at each place, 0 where fewer than m tokens came before."""
    h = np.zeros(len(s), dtype=np.uint64); x = s.astype(np.uint64); pw = np.uint64(1)
    for i in range(m):
        h[m - 1:] += x[m - 1 - i:len(s) - i] * pw; pw = pw * np.uint64(1000003)
    h ^= h >> np.uint64(29); h *= np.uint64(0xbf58476d1ce4e5b9); h ^= h >> np.uint64(32)
    h &= np.uint64(0xfffffffffffc0000); h[:m - 1] = 0
    return h


def table(s, m):
    """Sorted group keys with the top next token, its count and the group's total count."""
    q = keys(s[:-1], m); keep = q != 0
    pack = q[keep] | s[1:][keep].astype(np.uint64)
    u, cnt = np.unique(pack, return_counts=True); uk = u & np.uint64(0xfffffffffffc0000)
    first = np.flatnonzero(np.r_[True, uk[1:] != uk[:-1]])
    sc = (cnt.astype(np.uint64) << np.uint64(18)) | (u & np.uint64(0x3ffff))
    top = np.maximum.reduceat(sc, first); tot = np.add.reduceat(cnt, first)
    return uk[first], (top & np.uint64(0x3ffff)).astype(np.int64), (top >> np.uint64(18)).astype(np.int64), tot.astype(np.int64)


def kind(c, t):
    return np.searchsorted(C_EDGES, c, side="right") * 6 + np.searchsorted(P_EDGES, t / c, side="left")


def lookup(tables, s):
    """Hint token and kind at each place of s, one row per hint length, -1 for no hint."""
    T, K = [], []
    for (uk, t1, n1, tot), m in zip(tables, LV):
        q = keys(s, m); i = np.searchsorted(uk, q).clip(0, len(uk) - 1); hit = (uk[i] == q) & (q != 0)
        T.append(np.where(hit, t1[i], -1)); k = np.zeros(len(s), dtype=np.int64); k[hit] = kind(tot[i][hit], n1[i][hit]); K.append(k)
    return np.stack(T), np.stack(K)


def build_tables(tok=None):
    tok = tok or tokenizer(); A, B, C = load_pages(); path = os.path.join(HERE, "wiki", "tables.npz")
    if os.path.exists(path):
        z = np.load(path); return tok, [tuple(z[f"{m}_{i}"] for i in range(4)) for m in LV], A, B, C
    t0 = time.time(); s = np.concatenate(encode(tok, A)); print("tokens", len(s), round(time.time() - t0), "s", flush=True)
    tb = [table(s, m) for m in LV]; print("groups", [len(t[0]) for t in tb], round(time.time() - t0), "s", flush=True)
    np.savez(path, **{f"{m}_{i}": t[i] for m, t in zip(LV, tb) for i in range(4)})
    return tok, tb, A, B, C


def features():
    """Read pages with the model and keep, per place: the answer, its score, the total score, and each hint's score."""
    import mlx.core as mx
    from mlx_lm import load
    tok, tb, A, B, C = build_tables(); model, _ = load(os.path.join(HERE, "model"))
    for name, pages, want in (("trust", B, a.trust_pos), ("test", C, a.test_pos), ("fact", [r for r in A if "root" in r], a.fact_pos)):
        s = np.concatenate(encode(tok, pages[:3000])); rng = np.random.default_rng(1)
        starts = rng.permutation(len(s) - a.block - 1)[:want // a.block + 1]
        out = {k: [] for k in ("y", "ly", "lz", "tok", "kd", "lh", "top", "lt")}; t0 = time.time()
        for b in range(0, len(starts), a.batch):
            st = starts[b:b + a.batch]; X = np.stack([s[i:i + a.block] for i in st]); Y = np.stack([s[i + 1:i + a.block + 1] for i in st])
            lo = model(mx.array(X)).astype(mx.float32); lz = mx.logsumexp(lo, axis=-1); mx.eval(lo, lz); lo = np.array(lo, copy=False); lz = np.array(lz)
            for r in range(len(st)):
                ht, hk = lookup(tb, X[r]); pos = np.arange(8, a.block)  # the first words of a block have little context
                out["y"].append(Y[r, pos]); out["ly"].append(lo[r, pos, Y[r, pos]]); out["lz"].append(lz[r, pos]); out["top"].append(lo[r, pos].argmax(-1)); out["lt"].append(lo[r, pos].max(-1))
                out["tok"].append(ht[:, pos].T); out["kd"].append(hk[:, pos].T)
                out["lh"].append(np.stack([lo[r, pos, np.maximum(ht[L, pos], 0)] for L in range(len(LV))], 1))
            if b % (a.batch * 10) == 0:
                print(name, b, len(starts), round(time.time() - t0), "s", flush=True)
        np.savez(os.path.join(HERE, f"{name}.npz"), **{k: np.concatenate(v) for k, v in out.items()})
        print(name, "done", round(time.time() - t0), "s", flush=True)


def score(F, w, use_conf):
    """Loss with trust w, using only the saved numbers. p is each hinted token's share of the model's total score."""
    import torch
    n = len(LV); order = torch.arange(n)
    y, ly, lz, tk, kd, lh = (torch.as_tensor(F[k]) for k in ("y", "ly", "lz", "tok", "kd", "lh"))
    ly, lz, lh = ly.float(), lz.float(), lh.float(); mask = tk >= 0; p = (lh - lz[:, None]).clamp(max=0).exp()
    conf = torch.as_tensor(np.searchsorted(M_EDGES, p.numpy())) if use_conf else torch.zeros_like(kd)
    b = w[(kd + NK * order[None]) * NM + conf] * mask
    same = (tk[:, :, None] == tk[:, None, :]) & mask[:, :, None] & mask[:, None, :]  # hints that name one token add up
    tot = (same.float() * b[:, None, :]).sum(2); first = mask & ~(same & (order[None, :, None] > order[None, None, :])).any(2)
    R = 1 + (first.float() * p * (tot.exp() - 1)).sum(1); by = (first & (tk == y[:, None])).float().mul(tot).sum(1)
    return (lz - ly - by + R.log()).mean()


def right(F, w, use_conf):
    """At each place: is the top token the right one, with trust w."""
    import torch
    n = len(LV); y, tk, kd = F["y"], F["tok"], F["kd"]; lh, lz = F["lh"], F["lz"]
    p = np.exp(np.minimum(lh - lz[:, None], 0)); conf = np.searchsorted(M_EDGES, p) if use_conf else 0 * kd
    b = w.detach().numpy()[(kd + NK * np.arange(n)[None]) * NM + conf]; ok = np.zeros(len(y), dtype=bool)
    for i in range(len(y)):
        S = {}
        for L in range(n):
            if tk[i, L] >= 0:
                S[tk[i, L]] = S.get(tk[i, L], lh[i, L]) + b[i, L]
        best = max(S, key=S.get, default=None)
        pick = best if best is not None and S[best] > max(F["lt"][i], max((v for k, v in S.items() if k == F["top"][i]), default=-1e9)) else F["top"][i]
        ok[i] = pick == y[i]
    return ok


def accuracy(F, w, use_conf):
    return right(F, w, use_conf).mean()


def run_trust():
    import torch
    Tr, Te, Fa = (np.load(os.path.join(HERE, f"{n}.npz")) for n in ("trust", "test", "fact"))
    print("places", len(Tr["y"]), len(Te["y"]), len(Fa["y"]), "share with a hint", [round(float((Te["tok"][:, L] >= 0).mean()), 3) for L in range(len(LV))], flush=True)
    res = {}
    for use_conf in (0, 1):
        w = torch.zeros(len(LV) * NK * NM, requires_grad=True); opt = torch.optim.Adam([w], lr=.05)
        for i in range(a.steps):
            l = score(Tr, w, use_conf); opt.zero_grad(); l.backward(); opt.step()
        z = torch.zeros_like(w)
        for name, F_ in (("test", Te), ("fact", Fa)):
            off, on = score(F_, z, 0).item(), score(F_, w, use_conf).item()
            a0, a1 = accuracy(F_, z, 0), accuracy(F_, w, use_conf)
            res[f"{name}_{use_conf}"] = (off, on, a0, a1); print("conf" if use_conf else "kind", name, "loss off", round(off, 4), "on", round(on, 4), "top word off", round(a0, 4), "on", round(a1, 4), flush=True)
        np.save(os.path.join(HERE, f"trust{use_conf}.npy"), w.detach().numpy())
    json.dump(res, open(os.path.join(HERE, "trust_results.json"), "w"))


NAMES = "Alice Bob Carol David Emma Frank Grace Henry Irene Jack Laura Mark Nora Oscar Paula Peter Rachel Simon Tina Victor".split()


def comparison(n, rng):
    """Three names in a chain, statements in random order. The answer is the tallest or the shortest."""
    out = []
    for _ in range(n):
        x = list(rng.choice(NAMES, 3, replace=False)); st = [f"{x[0]} is taller than {x[1]}.", f"{x[1]} is taller than {x[2]}."]
        if rng.random() < .5:
            st.reverse()
        w, ans = ("tallest", x[0]) if rng.random() < .5 else ("shortest", x[2])
        out.append((" ".join(st) + f" The {w} person is", " " + ans))
    return out


def sums(n, rng):
    return [(f"{i} + {j} =", f" {i + j}") for i, j in rng.integers(10, 90, (n, 2))]


FACTS = [  # page, words as the page has them, the same fact in other words, answer
    ("2026 New Glenn rocket explosion", "On May 28, 2026, a New Glenn rocket operated by Blue Origin exploded during a static fire test at", "In May 2026, a New Glenn rocket exploded at a launch site called", "Cape"),
    ("2026 Grand Canyon flood", "On 29 August 2026, compound climate extremes and heavy rainfall caused a major flash flood that surged through", "The 2026 Grand Canyon flood surged through a creek called", "Bright"),
    ("Acquisition of Warner Bros. Discovery by Paramount Skydance", "On October 6, 2026, Paramount Skydance acquired Warner Bros. Discovery (WBD) for $", "Paramount Skydance paid for each Warner Bros. Discovery share a price of $", "31"),
    ("2026 Damascus bombings", "On 7 July 2026, two explosive devices exploded near the Four Seasons Hotel", "The 2026 bombs near a Four Seasons Hotel went off in the city of", "Damascus"),
    ("2026 Shabwah missile attack", "On 19 October 2026, a training center within a military camp in Shabwah Governorate controlled by the internationally recognized government of Yemen was struck by", "The October 2026 missile attack in Shabwah was carried out by the", "Houthi"),
    ("Apple A20 Pro", "It was announced on September 9, 2026, and is the successor to the Apple", "The chip that the Apple A20 Pro replaced is the Apple", "A19"),
    ("2026 Dallas apartment explosion", "On May 28, 2026, a natural gas explosion and five-alarm fire occurred at The Clyde, an apartment building in the Oak Cliff neighborhood of", "The Clyde apartment building that exploded in 2026 stands in the city of", "Dallas"),
    ("2 Valid IDs", "2 Valid IDs is a 2026 Philippine political drama film co-produced and directed by", "The director of the film 2 Valid IDs is", "Ma"),
    ("3 Weeks After", "3 Weeks After (Serbian: 3 nedelje posle) is a 2026 drama film directed by", "The director of the film 3 Weeks After is", "Miroslav"),
    ("5 More Minutes", "5 More Minutes (Spanish: 5 minutos m\u00e1s) is a 2026 Spanish fantasy comedy film directed by", "The director of the film 5 More Minutes is", "Javier"),
    ("2026 Beninese parliamentary election", "Parliamentary elections were held in Benin on 11 January 2026 to elect all", "In the January 2026 vote in Benin, the number of seats elected was", "109"),
    ("Patricia Baird", "Patricia Ann Baird,  (11 October 1937 \u2013 7 August 2026) was a British medical", "Patricia Baird, who died in August 2026, worked as a medical", "genetic"),
    ("2026 Romanian political crisis", "The 2026 Romanian political crisis was a constitutional and governmental crisis in Romania that began on 20 April 2026 and ended on", "The Romanian political crisis of 2026 finished on the date", "20"),
    ("2025\u20132026 Dutch cabinet formation", "After 117 days, the", "The Dutch cabinet formed after the October 2025 election is called the", "Jetten"),
    ("EU Kids Act", "The EU Kids Act is a proposed regulation by the European Commission and the European Union that would prohibit children under", "The EU Kids Act would ban social media accounts for children younger than", "13"),
    ("2026 Gay\u00e9ri attack", "On 30 June 2026, Burkinabe Forces repelled a large scale assault in", "The attack in Burkina Faso on 30 June 2026 took place in the town of", "Gay"),
    ("Acer Googlebook 14", "The Acer Googlebook 14 is a laptop developed by Acer Inc. and Google that runs", "The operating system on the Acer Googlebook 14 is", "Google"),
    ("2026 FIFA World Cup final halftime show", "The 2026 FIFA World Cup final halftime show was the halftime entertainment of the 2026 FIFA World Cup final, which took place on July 19, 2026, at", "The stadium that hosted the 2026 FIFA World Cup final is", "Met"),
]


def answer(model, tok, tb, w, q, n, on):
    """Greedy next tokens, with the trust that also sees how likely the model thinks each hinted token is."""
    import mlx.core as mx
    s = np.array(tok.encode(q, add_special_tokens=False).ids, dtype=np.int64); got = []
    for _ in range(n):
        lo = np.array(model(mx.array(s[None])).astype(mx.float32)[0, -1]); lz = np.logaddexp.reduce(lo)
        if on:
            ht, hk = lookup(tb, s); B = {}
            for L in range(len(LV)):
                t = ht[L, -1]
                if t >= 0:
                    B[t] = B.get(t, 0) + w[(hk[L, -1] + NK * L) * NM + np.searchsorted(M_EDGES, np.exp(min(lo[t] - lz, 0)))]
            for t, v in B.items():
                lo[t] += v
        got.append(int(lo.argmax())); s = np.r_[s, got[-1]]
    return tok.decode(got).strip()


def run_reason():
    from mlx_lm import load
    tok, tb, A, *_ = build_tables(); model, _ = load(os.path.join(HERE, "model")); res = {}
    w = np.load(os.path.join(HERE, "trust1.npy")); rng = np.random.default_rng(2); inA = {r["title"] for r in A}
    F = [x for x in FACTS if x[0] in inA]; print("facts with their page in the table", len(F), "of", len(FACTS), flush=True)
    for name, Q in (("facts same words", [(x[1], x[3]) for x in F]), ("facts other words", [(x[2], x[3]) for x in F]),
                    ("comparison", comparison(a.reason, rng)), ("sums", [(q, c.strip()) for q, c in sums(a.reason, rng)])):
        ok = [0, 0]; changed = 0; t0 = time.time()
        for q, c in Q:
            A0, A1 = (answer(model, tok, tb, w, q, {"comparison": 2, "sums": 4}.get(name, 3), on) for on in (0, 1))
            ok[0] += A0.strip().startswith(c.strip()); ok[1] += A1.strip().startswith(c.strip()); changed += A0 != A1
            if "facts" in name or ok[0] + changed < 4:
                print(" ", repr(q[-40:]), "off", repr(A0), "on", repr(A1), "want", repr(c), flush=True)
        res[name] = (ok[0], ok[1], changed, len(Q)); print(name, "right off", ok[0], "on", ok[1], "changed", changed, "of", len(Q), round(time.time() - t0), "s", flush=True)
    json.dump(res, open(os.path.join(HERE, "reason_results.json"), "w"))


if __name__ == "__main__":
    {"features": features, "trust": run_trust, "reason": run_reason}[a.mode]()
