"""Plans the v1.43 -> v1.44 number swap in the draft. Reads a text dump (lines P<n>|<text>), finds every number that
check_paper_v143.py would trace to the v1.43 ledger, and pairs it with the same position in the same row of the v1.44 ledger
(rows matched by section and name, build-measured numbers stripped from the name).

Writes <out>.tsv: paragraph, offset in paragraph, old text, new text, status, row(s), context. Status:
  same       the number is unchanged in v1.44
  change     one row, one new value
  ambiguous  the number belongs to rows that disagree in v1.44, or appears more than once in a row: needs a person
"""
import csv
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
A = json.load(open(ROOT / "campaign_v143/final/analysis/v143_results.json"))["rows"]
B = json.load(open(ROOT / "campaign_v144/final/analysis/v144_results.json"))["rows"]

def norm(k):
    k = re.sub(r",? ?arg\(k\) -?[\d.]+ deg", "", k); k = re.sub(r" \(\d+ loads\)", "", k)
    k = re.sub(r" L_50 \([\d.]+\)", " L_50", k); k = re.sub(r", rounded, \d+ ns group", ", rounded", k)
    return re.sub(r"SCALING_OUT=resource_scaling_v14\d", "SCALING_OUT=resource_scaling", k)

BK = {(r["section"], norm(r["key"])): r for r in B}
LED = {}            # v1.43 number (no commas) -> list of (row label, v1.44 text or None)
for r in A:
    b = BK.get((r["section"], norm(r["key"])))
    for i, t in enumerate(r["text"]):
        new = None
        if b is not None and len(b["text"]) == len(r["text"]):
            new = str(b["text"][i])
        dup = sum(str(x).replace(",", "") == str(t).replace(",", "") for x in r["text"]) > 1
        LED.setdefault(str(t).replace(",", ""), []).append((f"{r['section']}: {r['key']}"[:90], new, dup, str(t)))

NUM = re.compile(r"(?<![\w.])[-−+±]?\d[\d,]*(?:\.\d+)?(?![\w])")
CITE = re.compile(r"\[\d{1,2}\]")
rows = []
for ln in open(sys.argv[1], encoding="utf-8"):
    m = re.match(r"P(\d+)\|(.*)", ln.rstrip("\n"))
    if not m or re.match(r"\[\d+\] ", m[2]):
        continue
    n, t = int(m[1]), m[2]
    if t == "References":
        break
    t2 = CITE.sub(lambda q: " " * len(q[0]), t)
    for q in NUM.finditer(t2):
        lead = len(q[0]) - len(q[0].lstrip("-−+±"))
        tok = q[0][lead:]
        if tok.endswith("."):
            tok = tok[:-1]
        raw = tok.replace(",", "")
        if len(raw) == 1 or re.match(r"^(19|20)\d\d$", raw) or raw not in LED:
            continue
        c = LED[raw]
        news = {x[1] for x in c}
        if any(x[2] for x in c) or None in news or len(news) > 1:
            st, new = "ambiguous", " / ".join(sorted(str(x) for x in news))
        else:
            new = news.pop()
            # keep the draft's own comma style
            new = new if ("," in tok) == ("," in new) else (new.replace(",", "") if "," not in tok else f"{float(new):,.{len(new.split('.')[1]) if '.' in new else 0}f}")
            st = "same" if new.replace(",", "") == raw else "change"
        s0 = q.start() + lead
        rows.append([f"P{n}", s0, tok, new, st, " || ".join(x[0] for x in c), t[max(0, s0 - 60):s0 + len(tok) + 40]])
with open(sys.argv[2], "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["para", "offset", "old", "new", "status", "rows", "context"])
    w.writerows(rows)
from collections import Counter
print(Counter(r[4] for r in rows), "paragraphs touched:", len({r[0] for r in rows if r[4] != "same"}))
