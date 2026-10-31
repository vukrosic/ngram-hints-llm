"""Where do hints help? Each setting builds a table, trains trust, and scores unseen files of that kind of text.

python domains.py          every setting
python domains.py docs     own documents only
"""
import glob, json, os, site, sys, time
import numpy as np
ONLY = sys.argv[1:]
sys.argv = ["hints.py", "x"]
import hints as h
import mlx.core as mx, torch
from mlx_lm import load

HERE = h.HERE; TRUST, TEST, BLOCK, BATCH = 10000, 8000, 128, 8
SITE = site.getsitepackages()[0] + "/"; DOCS = os.environ.get("DOCS", ".")  # DOCS: a folder of your own Markdown files
tok = h.tokenizer(); model, _ = load(os.path.join(HERE, "model"))


def files(pattern, skip=(), cap=30_000_000):
    P = [p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p) and not any(x in p for x in skip)]
    np.random.default_rng(0).shuffle(P); out = []; n = 0; seen = set()
    for p in P:
        x = open(p, errors="ignore").read()
        if len(x) > 500 and n < cap and x not in seen:  # exact copies of a file count once
            seen.add(x); out.append(dict(title=os.path.basename(p), text=x)); n += len(x)
    return out


def join(pages):
    return np.concatenate(h.encode(tok, pages))


def read(s, tb, want, BLOCK=BLOCK, BATCH=BATCH, last=None):
    """Same saved numbers as hints.features, for one stretch of text. last keeps only the last places of each block."""
    rng = np.random.default_rng(1); last = last or BLOCK - 8; starts = rng.permutation(len(s) - BLOCK - 1)[:want // last + 1]
    out = {k: [] for k in ("y", "ly", "lz", "tok", "kd", "lh", "top", "lt")}; pos = np.arange(BLOCK - last, BLOCK)
    for b in range(0, len(starts), BATCH):
        st = starts[b:b + BATCH]; X = np.stack([s[i:i + BLOCK] for i in st]); Y = np.stack([s[i + 1:i + BLOCK + 1] for i in st])
        lo = model(mx.array(X)).astype(mx.float32); lz = mx.logsumexp(lo, axis=-1); mx.eval(lo, lz); lo = np.array(lo, copy=False); lz = np.array(lz)
        for r in range(len(st)):
            ht, hk = h.lookup(tb, X[r])
            out["y"].append(Y[r, pos]); out["ly"].append(lo[r, pos, Y[r, pos]]); out["lz"].append(lz[r, pos]); out["top"].append(lo[r, pos].argmax(-1)); out["lt"].append(lo[r, pos].max(-1))
            out["tok"].append(ht[:, pos].T); out["kd"].append(hk[:, pos].T)
            out["lh"].append(np.stack([lo[r, pos, np.maximum(ht[L, pos], 0)] for L in range(len(h.LV))], 1))
    return {k: np.concatenate(v) for k, v in out.items()}


def run(name, table_pages, trust_pages, test_pages, **kw):
    t0 = time.time(); S = join(table_pages); tb = [h.table(S, m) for m in h.LV]; t1 = time.time()
    Tr, Te = read(join(trust_pages), tb, TRUST, **kw), read(join(test_pages), tb, TEST, **kw); t2 = time.time()
    w = torch.zeros(len(h.LV) * h.NK * h.NM, requires_grad=True); opt = torch.optim.Adam([w], lr=.05)
    for _ in range(300):
        l = h.score(Tr, w, 1); opt.zero_grad(); l.backward(); opt.step()
    z = torch.zeros_like(w); off, on = h.score(Te, z, 0).item(), h.score(Te, w, 1).item(); a0, a1 = h.accuracy(Te, z, 0), h.accuracy(Te, w, 1)
    R = dict(name=name, table_tokens=len(S), loss_off=round(off, 3), loss_on=round(on, 3), top_off=round(float(a0), 3), top_on=round(float(a1), 3), hit6=round(float((Te["tok"][:, 3] >= 0).mean()), 3), hit3=round(float((Te["tok"][:, 1] >= 0).mean()), 3), table_s=round(t1 - t0), read_s=round(t2 - t1), trust_s=round(time.time() - t2))
    print(json.dumps(R), flush=True); open(os.path.join(HERE, "domain_results.jsonl"), "a").write(json.dumps(R) + "\n")
    return tb, w.detach().numpy()


def parts(P):
    n = len(P); return P[:int(n * .7)], P[int(n * .7):int(n * .85)], P[int(n * .85):]


if __name__ == "__main__":
    md = files(DOCS + "/**/*.md", skip=("node_modules", "/.git/", "site-packages"))
    A, B, C = parts(md); run("own documents", A, B, C)
    if ONLY:
        sys.exit()
    tf = files(SITE + "transformers/**/*.py"); th = files(SITE + "torch/**/*.py")
    A, B, C = parts(tf); run("code same library", A, B, C)
    run("code other library", tf, th[:len(th) // 2], th[len(th) // 2:])
    A, B, C = h.load_pages()
    for f in (.01, .1):
        run(f"wikipedia {f:.0%} of table", A[:int(len(A) * f)], B[:3000], C[:3000])

    # Saved Wikipedia test: where does the gain sit?
    Te = np.load(os.path.join(HERE, "test.npz")); w = torch.tensor(np.load(os.path.join(HERE, "trust1.npy"))); z = torch.zeros_like(w)
    r0, r1 = h.right(Te, z, 0), h.right(Te, w, 1); longest = (Te["tok"] >= 0).sum(1); conf = np.exp(Te["lt"] - Te["lz"])
    print("hints firing | share of places | top off | top on")
    for k in range(5):
        m = longest == k
        print(k, round(float(m.mean()), 3), round(float(r0[m].mean()), 3), round(float(r1[m].mean()), 3))
    print("model confidence in its top word | share | top off | top on")
    for lo_, hi in ((0, .1), (.1, .25), (.25, .5), (.5, .9), (.9, 1.01)):
        m = (conf >= lo_) & (conf < hi)
        print(lo_, hi, round(float(m.mean()), 3), round(float(r0[m].mean()), 3), round(float(r1[m].mean()), 3))
