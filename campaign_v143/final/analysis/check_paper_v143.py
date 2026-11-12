"""Compares every number in a text dump of the draft with the v1.43 ledger. Lines of the dump are P<n>|<text>.

Each number gets one verdict, in this order:
  v1.43      in the ledger (a coincidence is possible, so --all prints the ledger row for a human to read)
  EXEMPT     listed in paper_notes/v143_exempt.tsv: paragraph, number, class, reason. Every exemption is written down
  STALE      equals a value from the v1.40/v1.41 results and is not in the ledger: the likely slip
  CITED      in a sentence with a citation: taken to be someone else's number
  OPEN       listed as needing a person (class "open" in the exemptions file); submission needs it empty
  UNCLASSED  none of these; submission needs it empty

A build name other than v1.43 is reported separately. Limits: it reads text only (the numbers inside figure and table
images have to be checked through their generators), and a number can match the ledger by chance.

"""
import csv
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
L = json.load(open(ROOT / "campaign_v143/final/analysis/v143_results.json"))
LED = {}
for r in L["rows"]:
    for t in r["text"]:
        LED.setdefault(t.replace(",", ""), []).append(f"{r['section']}: {r['key']}")

def collect(o, out):
    if isinstance(o, dict):
        for v in o.values():
            collect(v, out)
    elif isinstance(o, list):
        for v in o:
            collect(v, out)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        for d in (1, 2, 3):
            out.add(f"{abs(o):.{d}f}")
    return out

OLD = set()
for f in ("campaign_v140L1/v141_100k_stats.json", "campaign_v140L1/v141_100k_swap_stats.json", "campaign_v140L1/A_v141_live_headline_19400.json",
          "audit_fixes/R18/r18_inloop_stats.json", "audit_fixes/R18_full/inloop_full_stats.json", "audit_fixes/R3_R2/r3_headlines.json", "audit_fixes/R3_R2/r3_linear.json"):
    collect(json.load(open(ROOT / f)), OLD)
for v in ("95.53", "95.51", "1352", "137484", "197433", "0.213", "291", "107.8", "45.6", "0.78", "0.715"):
    OLD.add(v)
# small round values are not evidence of anything
OLD = {v for v in OLD if len(v.replace(".", "")) >= 3 and float(v) >= 1 or v in ("0.213", "0.78", "0.715", "0.64", "0.85", "0.69", "0.95")}

EX = []
xp = ROOT / "paper_notes/v143_exempt.tsv"
if xp.exists():
    for r in csv.reader(open(xp, encoding="utf-8"), delimiter="\t"):
        if r and not r[0].startswith("#"):
            EX.append(r)
exempt = {(r[0], r[1].lstrip("-−+±").replace(",", "").rstrip(".")): r[2] for r in EX}

NUM = re.compile(r"(?<![\w.])[-−+±]?\d[\d,]*(?:\.\d+)?(?![\w])")
CITE = re.compile(r"\[\d{1,2}\]")
paras = []
for ln in open(sys.argv[1], encoding="utf-8"):
    m = re.match(r"P(\d+)\|(.*)", ln.rstrip("\n"))
    if m and not re.match(r"\[\d+\] ", m[2]):
        paras.append((int(m[1]), m[2]))
if paras and any(t == "References" for _, t in paras):
    stop = next(n for n, t in paras if t == "References")
    paras = [(n, t) for n, t in paras if n < stop]

count = Counter(); lines = {"STALE": [], "UNCLASSED": [], "v1.43": [], "OPEN": []}
for n, t in paras:
    for b in sorted(set(re.findall(r"v1\.\d+", t))):
        if b != "v1.43":
            lines.setdefault("OTHER BUILD", []).append(f"P{n}: {b}")
            count["OTHER BUILD"] += 1
    for s in re.split(r"(?<=[.;:])\s+", t):
        s2 = CITE.sub(lambda m: " " * len(m[0]), s)
        for m in NUM.finditer(s2):
            raw = m[0].lstrip("-−+±").replace(",", "")
            if raw.endswith("."):
                raw = raw[:-1]
            if len(raw) == 1 or re.match(r"^(19|20)\d\d$", raw):   # single digits and years carry no result
                count["trivial"] += 1
                continue
            if raw in LED:
                v = "v1.43"
            elif (f"P{n}", raw) in exempt or ("*", raw) in exempt:
                v = "OPEN" if exempt.get((f"P{n}", raw)) == "open" else "EXEMPT"
            elif raw in OLD:
                v = "STALE"
            elif CITE.search(s):
                v = "CITED"
            else:
                v = "UNCLASSED"
            count[v] += 1
            if v in lines:
                extra = f"  <- {LED[raw][0]}" if v == "v1.43" else ""
                lines[v].append(f"P{n}: {m[0].strip()}  | {s[max(0, m.start() - 45):m.end() + 35]}{extra}")

ver = subprocess.run([sys.executable, str(ROOT / "campaign_v143/final/analysis/make_v143_ledger.py"), "--verify"], capture_output=True, text=True).stdout.strip()
print(ver)
print("counts:", dict(count))
for k in ("OTHER BUILD", "STALE", "UNCLASSED", "OPEN") + (("v1.43",) if "--all" in sys.argv else ()):
    print(f"\n== {k} ({len(lines.get(k, []))})")
    for x in lines.get(k, []):
        print(" ", x)
sys.exit(0 if not (count["STALE"] or count["UNCLASSED"] or count["OTHER BUILD"] or count["OPEN"]) else 1)
