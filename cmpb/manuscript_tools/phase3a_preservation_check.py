#!/usr/bin/env python3
"""Phase-3A preservation check: reference build (V7) against this build (V8).
Usage: python3 tools/phase3a_preservation_check.py REF_DIR [OUT.md]
Checks: (1) data/verification files and stored plot-number files byte-identical; (2) abstract/front matter, declarations,
discussion, background and theory segments identical; (3) theorem-like environments and Algorithm 1 identical;
(4) footnotes identical; (5) multi-digit numbers in the compiled PDFs (set difference); (6) figure numbering and unresolved references."""
import hashlib, os, re, subprocess, sys, collections
ref = os.path.abspath(sys.argv[1]); R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(R, "PHASE3A_PRESERVATION_CHECK.md")
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def tree(d, sub): 
    res = {}
    for root, _, fs in os.walk(os.path.join(d, sub)):
        for f in fs: res[os.path.relpath(os.path.join(root, f), d)] = sha(os.path.join(root, f))
    return res
L = ["# Phase 3A preservation check (V7 -> V8)\n"]; ok = True
# (1) data
for sub in ("data", "verify_r0"):
    a, b = tree(ref, sub), tree(R, sub); diff = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    L.append(f"\n**`{sub}/` files.** {len(a)} in V7, {len(b)} in V8; byte-identical: {len(a) - len(diff)}; differing: {diff}\n"); ok &= not diff
nums = [f for f in ("fig_numbers.json", "fig_p4_exploratory_numbers.json", "fig_frontier_gpu_numbers.json") ]
for f in nums:
    same = sha(os.path.join(ref, "paper/figures", f)) == sha(os.path.join(R, "paper/figures", f)); ok &= same
    L.append(f"- plotted-number file `{f}` identical: {same}\n")
for f in sorted(os.listdir(os.path.join(ref, "paper/figures"))):
    if not f.endswith(".pdf"): continue
    same = sha(os.path.join(ref, "paper/figures", f)) == sha(os.path.join(R, "paper/figures", f)); ok &= same
    L.append(f"- figure file `{f}` byte-identical to V7: {same}\n")
# (2) segments
L.append("\n**Segments.**\n")
for f in sorted(os.listdir(os.path.join(ref, "seg"))):
    if not f.endswith(".tex"): continue
    pa, pb = os.path.join(ref, "seg", f), os.path.join(R, "seg", f)
    if not os.path.exists(pb): L.append(f"- `{f}`: REMOVED\n"); ok = False; continue
    same = sha(pa) == sha(pb); L.append(f"- `{f}`: {'identical' if same else 'changed (see PHASE3A_RESULTS_CHANGELOG.md)'}\n")
    if f in ("30_background.tex", "50_theory.tex", "95_declarations.tex", "99_bib.tex", "20_intro.tex"): ok &= same
# (3) theorem-like environments and algorithm
ENV = re.compile(r"\\begin\{(theorem|lemma|proposition|definition|algorithm)\}(.*?)\\end\{\1\}", re.S)
def stm(t): return [re.sub(r"\s+", " ", re.sub(r"%.*", "", m.group(2))).strip() for m in ENV.finditer(t)]
for n in ("CMPB_REVISED_MAIN", "CMPB_REVISED_SUPPLEMENT"):
    a, b = stm(open(os.path.join(ref, n + ".tex"), encoding="utf-8").read()), stm(open(os.path.join(R, n + ".tex"), encoding="utf-8").read())
    L.append(f"\n**{n}: theorem-like environments and algorithms.** V7 {len(a)}, V8 {len(b)}; identical: {a == b}\n"); ok &= a == b
# (4) footnotes
FN = re.compile(r"\\footnote\{")
for n in ("CMPB_REVISED_MAIN", "CMPB_REVISED_SUPPLEMENT"):
    a, b = (len(FN.findall(open(os.path.join(d, n + ".tex"), encoding="utf-8").read())) for d in (ref, R))
    L.append(f"- footnotes in {n}: V7 {a}, V8 {b}\n"); ok &= a == b
# (5) numbers
def pdftext(d, n): return subprocess.run(["pdftotext", os.path.join(d, n + ".pdf"), "-"], capture_output=True, text=True).stdout
def big(t): return set(x for x in re.findall(r"\d[\d,]*(?:\.\d+)?%?", t.replace("\u2212", "-")) if len(re.sub(r"\D", "", x)) >= 2)
for n in ("CMPB_REVISED_MAIN", "CMPB_REVISED_SUPPLEMENT"):
    a, b = big(pdftext(ref, n)), big(pdftext(R, n))
    L.append(f"\n**{n}: multi-digit numbers (PDF text).** absent in V8: {sorted(a - b)}; new in V8: {sorted(b - a)}\n")
# (6) figure numbering
def caps(t, kind): return [re.match(rf"^({kind} S?\d+)", l).group(1) for l in t.splitlines() if re.match(rf"^{kind} S?\d+:", l)]
for n, pre in (("CMPB_REVISED_MAIN", "Figure"), ("CMPB_REVISED_SUPPLEMENT", "Figure")):
    t = pdftext(R, n); c = caps(t, pre); idx = [int(re.sub(r"\D", "", x)) for x in c]
    cont = idx == list(range(1, len(idx) + 1)); ok &= cont
    L.append(f"\n**{n}: figure captions in PDF order** {c}; consecutive numbering: {cont}; unresolved '??': {t.count('??')}\n"); ok &= t.count("??") == 0

# (4b) footnote texts
def fn_texts(t):
    out_, i = [], 0
    while True:
        i = t.find("\\footnote{", i)
        if i < 0: break
        k, d = i + len("\\footnote{"), 1
        while d and k < len(t):
            d += (t[k] == "{") - (t[k] == "}"); k += 1
        out_.append(re.sub(r"\s+", " ", t[i:k])); i = k
    return out_
for n in ("CMPB_REVISED_MAIN", "CMPB_REVISED_SUPPLEMENT"):
    a, b = (fn_texts(open(os.path.join(d, n + ".tex"), encoding="utf-8").read()) for d in (ref, R))
    miss = [x[:70] for x in a if x not in b]
    L.append(f"- footnote texts of {n} present unchanged in V8: {len(a) - len(miss)}/{len(a)}; changed or missing: {miss}\n"); ok &= not miss
# (5b) numbers removed from the main PDF must still exist in the V8 main text or supplement
mA, mB = big(pdftext(ref, "CMPB_REVISED_MAIN")), big(pdftext(R, "CMPB_REVISED_MAIN"))
sB = big(pdftext(R, "CMPB_REVISED_SUPPLEMENT")); gone = sorted((mA - mB) - sB)
L.append(f"\n**Numbers removed from the main text and absent from the V8 supplement:** {gone}\n")
# (6b) first-reference order of main floats
tex = open(os.path.join(R, "CMPB_REVISED_MAIN.tex"), encoding="utf-8").read()
aux = open(os.path.join(R, "CMPB_REVISED_MAIN.aux"), encoding="utf-8", errors="replace").read()
labs = [(int(m.group(2)), m.group(1)) for m in re.finditer(r"\\newlabel\{((?:fig|tab):[^}@]+)\}\{\{(\d+)\}", aux)]
for kind in ("fig", "tab"):
    ls = [l for _, l in sorted(set(labs)) if l.startswith(kind + ":")]
    pos = []
    for l in ls:
        m = [x.start() for x in re.finditer(r"\\[cC]ref\{([^}]*)\}", tex) if l in x.group(1).split(",")]
        pos.append(m[0] if m else -1)
    good = all(pos[i] <= pos[i + 1] for i in range(len(pos) - 1) if pos[i] >= 0 and pos[i + 1] >= 0)
    L.append(f"- first-reference order of main {kind} labels follows numbering: {good}; unreferenced: {[l for l, p in zip(ls, pos) if p < 0]}\n"); ok &= good
L.append(f"\n**Overall (automatic part): {'PASS' if ok else 'CHECK'}**\n")
open(out, "w", encoding="utf-8").write("".join(L)); print("written", out, "PASS" if ok else "CHECK")
