#!/usr/bin/env python3
"""Phase-1 preservation check: compare a reference build (V5) with this build (V6).
Usage: python3 tools/phase1_preservation_check.py REF_DIR [OUT.md]
Reports (1) numeric tokens removed/added in the compiled PDFs, (2) theorem-like environments (statements)
that changed, (3) labels removed/added, (4) figure/table caption order and numbering."""
import re, sys, subprocess, collections, os
ref = os.path.abspath(sys.argv[1]); R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(R, "PHASE1_PRESERVATION_CHECK.md")
DOCS = {"main": "CMPB_REVISED_MAIN", "supplement": "CMPB_REVISED_SUPPLEMENT"}
def pdftext(d, n): return subprocess.run(["pdftotext", os.path.join(d, n + ".pdf"), "-"], capture_output=True, text=True).stdout
def numbers(t):
    t = t.replace("\u2212", "-")
    return collections.Counter(re.findall(r"\d[\d,]*(?:\.\d+)?%?", re.sub(r"\s+", " ", t)))
def read(d, n): return open(os.path.join(d, n + ".tex"), encoding="utf-8").read()
ENV = re.compile(r"\\begin\{(theorem|lemma|proposition|definition)\}(.*?)\\end\{\1\}", re.S)
def statements(t):
    res = {}
    for m in ENV.finditer(t):
        body = m.group(2); lab = re.search(r"\\label\{([^}]*)\}", body)
        norm = re.sub(r"\s+", " ", re.sub(r"%.*", "", body)).strip()
        res[lab.group(1) if lab else norm[:40]] = norm
    return res
def labels(t): return set(re.findall(r"\\label\{([^}]*)\}", t))
def captions(txt, kind):
    return [re.sub(r"\s+", " ", l)[:32] for l in txt.splitlines() if re.match(rf"^{kind} (S?\d+)[:.]", l)]
L = ["# Phase 1 preservation check (V5 -> V6)\n"]
allok = True
for key, n in DOCS.items():
    L.append(f"\n## {key}\n")
    a, b = numbers(pdftext(ref, n)), numbers(pdftext(R, n))
    rem = {k: a[k] - b[k] for k in a if a[k] > b[k]}; add = {k: b[k] - a[k] for k in b if b[k] > a[k]}
    L.append(f"**Numeric tokens in the compiled PDF.** V5: {sum(a.values())}, V6: {sum(b.values())}. "
             f"Tokens with fewer occurrences in V6: {len(rem)}; with more occurrences: {len(add)}.\n")
    L.append("Fewer in V6 (token: decrease): " + ", ".join(f"`{k}`:{v}" for k, v in sorted(rem.items())) + "\n")
    L.append("More in V6 (token: increase): " + ", ".join(f"`{k}`:{v}" for k, v in sorted(add.items())) + "\n")
    ta, tb = read(ref, n), read(R, n)
    sa, sb = statements(ta), statements(tb)
    same = [k for k in sa if k in sb and sa[k] == sb[k]]
    changed = [k for k in sa if k in sb and sa[k] != sb[k]]; gone = [k for k in sa if k not in sb]; new = [k for k in sb if k not in sa]
    L.append(f"**Theorem-like environments (definition, theorem, lemma, proposition).** identical: {len(same)}; changed: {len(changed)} {changed}; "
             f"removed: {gone}; added: {new}.\n")
    if changed or gone: allok = False
    def big(t):
        return set(x for x in re.findall(r"\d[\d,]*(?:\.\d+)?%?", t.replace("\u2212", "-")) if len(re.sub(r"\D", "", x)) >= 2)
    L.append(f"**Multi-digit numbers (set difference).** absent in V6: {sorted(big(pdftext(ref, n)) - big(pdftext(R, n)))}; new in V6: {sorted(big(pdftext(R, n)) - big(pdftext(ref, n)))}.\n")
    la, lb = labels(ta), labels(tb)
    L.append(f"**Labels.** removed: {sorted(la - lb)}; added: {sorted(lb - la)}.\n")
    pa, pb = pdftext(ref, n), pdftext(R, n)
    for kind in ("Figure", "Table"):
        ca, cb = captions(pa, kind), captions(pb, kind)
        L.append(f"**{kind} captions (PDF order).** V5: {len(ca)}, V6: {len(cb)}. " + ("Figure order and numbering unchanged.\n" if kind == "Figure" and ca == cb else "\n"))
        if kind == "Figure" and ca != cb: allok = False; L.append("  - DIFFERENCE in figure captions\n")
open(out, "w", encoding="utf-8").write("".join(L))
print("written", out, "; theorem-like statements unchanged:", allok)
