"""Download a small sample of 2026 Wikipedia pages as plain text.

python fetch.py --pages 600

Pages come from the 2026 categories, so most of them cover events newer than the model's training text.
"""
import argparse, json, os, re, time, urllib.error, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--pages", type=int, default=600)
a = p.parse_args()
API = "https://en.wikipedia.org/w/api.php"
UA = {"User-Agent": "ngram-hints-research/0.1 (small sample for a language model test)"}
ROOTS = ["2026 in science", "2026 in sports", "2026 in politics", "2026 in film", "2026 in music", "2026 in spaceflight",
         "2026 elections", "2026 disasters", "2026 in technology", "2026 in law", "2026 in economics", "2026 deaths"]


def get(**q):
    q.update(format="json", formatversion=2)
    r = urllib.request.Request(API + "?" + urllib.parse.urlencode(q), headers=UA)
    for _ in range(8):
        try:
            d = json.load(urllib.request.urlopen(r, timeout=30)); time.sleep(1.5)
            return d
        except urllib.error.HTTPError as e:
            time.sleep(int(e.headers.get("Retry-After") or 30))
        except Exception:
            time.sleep(5)
    return {}


def members(cat, kind, n):
    d = get(action="query", list="categorymembers", cmtitle="Category:" + cat, cmtype=kind, cmlimit=min(n, 500))
    return [m["title"] for m in d.get("query", {}).get("categorymembers", [])]


titles, seen = [], set()
per = a.pages // len(ROOTS) * 3
for root in ROOTS:
    got = members(root, "page", per)
    for sub in members(root, "subcat", 30):
        if len(got) >= per:
            break
        got += members(sub.split(":", 1)[1], "page", per - len(got))
    for t in got:
        if t not in seen and not t.startswith(("List of", "Template:", "Category:", "Portal:")):
            seen.add(t); titles.append((root, t))
    print(root, len(got), len(titles), flush=True)
    time.sleep(.2)

# Take pages evenly across the roots.
by = {}
for r, t in titles:
    by.setdefault(r, []).append(t)
pick = []
while len(pick) < a.pages and any(by.values()):
    for r in ROOTS:
        if by.get(r) and len(pick) < a.pages:
            pick.append((r, by[r].pop(0)))

path = os.path.join(HERE, "pages.jsonl"); old = {json.loads(l)["title"] for l in open(path)} if os.path.exists(path) else set()
out = open(path, "a"); t0 = time.time(); n = len(old)
for i, (r, t) in enumerate(pick):
    if t in old:
        continue
    d = get(action="query", prop="extracts", explaintext=1, titles=t, redirects=1)
    pg = d.get("query", {}).get("pages", [{}])[0]; x = pg.get("extract", "")
    x = re.split(r"\n== (See also|References|External links|Notes|Further reading|Citations) ==", x)[0]
    if len(x) > 800:
        out.write(json.dumps(dict(root=r, title=pg.get("title", t), text=x)) + "\n"); out.flush(); n += 1
    if i % 100 == 0:
        print(i, n, round(time.time() - t0), flush=True)
print("pages", n, "seconds", round(time.time() - t0))
