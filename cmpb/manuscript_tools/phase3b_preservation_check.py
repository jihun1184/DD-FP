#!/usr/bin/env python3
"""Phase-3B preservation check: reference build (V8) against this build (V9).
Usage: python3 tools/phase3b_preservation_check.py REF_DIR [OUT.md]
Checks: (1) data/verification files and stored plot-number files byte-identical; (2) background, introduction, method and preamble segments identical (front, results, theory, discussion, declarations and bibliography are expected to change in 3B); (3) theorem-like environments and Algorithm 1 identical;
(4) footnotes identical; (5) multi-digit numbers in the compiled PDFs (set difference); (6) figure numbering and unresolved references."""
import hashlib, os, re, subprocess, sys, collections
ref = os.path.abspath(sys.argv[1]); R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(R, "PHASE3B_PRESERVATION_CHECK.md")
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def tree(d, sub): 
    res = {}
    for root, _, fs in os.walk(os.path.join(d, sub)):
        for f in fs: res[os.path.relpath(os.path.join(root, f), d)] = sha(os.path.join(root, f))
    return res
L = ["# Phase 3B preservation and submission-form check (V8 -> V9)\n"]; ok = True
# (1) data
for sub in ("data", "verify_r0"):
    a, b = tree(ref, sub), tree(R, sub); diff = [k for k in sorted(a) if a.get(k) != b.get(k)]; added = sorted(set(b) - set(a))
    L.append(f"\n**`{sub}/` files.** {len(a)} in V8, {len(b)} in V9; V8 files byte-identical in V9: {len(a) - len(diff)}; changed or removed: {diff}; added in V9: {added}\n"); ok &= not diff
nums = [f for f in ("fig_numbers.json", "fig_p4_exploratory_numbers.json", "fig_frontier_gpu_numbers.json") ]
for f in nums:
    same = sha(os.path.join(ref, "paper/figures", f)) == sha(os.path.join(R, "paper/figures", f)); ok &= same
    L.append(f"- plotted-number file `{f}` identical: {same}\n")
for f in sorted(os.listdir(os.path.join(ref, "paper/figures"))):
    if not f.endswith(".pdf"): continue
    same = sha(os.path.join(ref, "paper/figures", f)) == sha(os.path.join(R, "paper/figures", f)); ok &= same
    L.append(f"- figure file `{f}` byte-identical to V8: {same}\n")
# (2) segments
L.append("\n**Segments.**\n")
for f in sorted(os.listdir(os.path.join(ref, "seg"))):
    if not f.endswith(".tex"): continue
    pa, pb = os.path.join(ref, "seg", f), os.path.join(R, "seg", f)
    if not os.path.exists(pb): L.append(f"- `{f}`: REMOVED\n"); ok = False; continue
    same = sha(pa) == sha(pb); L.append(f"- `{f}`: {'identical' if same else 'changed (see PHASE3B_FINAL_CHANGELOG.md)'}\n")
    if f in ("30_background.tex", "20_intro.tex", "00_preamble.tex"): ok &= same
# (3) theorem-like environments and algorithm
ENV = re.compile(r"\\begin\{(theorem|lemma|proposition|definition|algorithm)\}(.*?)\\end\{\1\}", re.S)
def stm(t): return [re.sub(r"\s+", " ", re.sub(r"%.*", "", m.group(2))).strip() for m in ENV.finditer(t)]
for n in ("CMPB_REVISED_MAIN", "CMPB_REVISED_SUPPLEMENT"):
    a, b = stm(open(os.path.join(ref, n + ".tex"), encoding="utf-8").read()), stm(open(os.path.join(R, n + ".tex"), encoding="utf-8").read())
    L.append(f"\n**{n}: theorem-like environments and algorithms.** V8 {len(a)}, V9 {len(b)}; identical: {a == b}\n"); ok &= a == b
# (4) footnotes
FN = re.compile(r"\\footnote\{")
for n in ("CMPB_REVISED_MAIN", "CMPB_REVISED_SUPPLEMENT"):
    a, b = (len(FN.findall(open(os.path.join(d, n + ".tex"), encoding="utf-8").read())) for d in (ref, R))
    L.append(f"- footnotes in {n}: V8 {a}, V9 {b}\n"); ok &= a == b
# (5) numbers
def pdftext(d, n): return subprocess.run(["pdftotext", os.path.join(d, n + ".pdf"), "-"], capture_output=True, text=True).stdout
def big(t): return set(x for x in re.findall(r"\d[\d,]*(?:\.\d+)?%?", t.replace("\u2212", "-")) if len(re.sub(r"\D", "", x)) >= 2)
for n in ("CMPB_REVISED_MAIN", "CMPB_REVISED_SUPPLEMENT"):
    a, b = big(pdftext(ref, n)), big(pdftext(R, n))
    L.append(f"\n**{n}: multi-digit numbers (PDF text).** absent in V9: {sorted(a - b)}; new in V9: {sorted(b - a)}\n")
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
    L.append(f"- footnote texts of {n} present unchanged in V9: {len(a) - len(miss)}/{len(a)}; changed or missing: {miss}\n"); ok &= not miss
# (5b) numbers removed from the main PDF must still exist in the V9 main text or supplement
mA, mB = big(pdftext(ref, "CMPB_REVISED_MAIN")), big(pdftext(R, "CMPB_REVISED_MAIN"))
sB = big(pdftext(R, "CMPB_REVISED_SUPPLEMENT")); gone = sorted((mA - mB) - sB)
L.append(f"\n**Numbers removed from the main text and absent from the V9 supplement:** {gone}\n")
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

# (7) Phase-3B form checks
srcs = {f: open(os.path.join(R, "seg", f), encoding="utf-8").read() for f in os.listdir(os.path.join(R, "seg")) if f.endswith(".tex")}
supp = open(os.path.join(R, "CMPB_REVISED_SUPPLEMENT.tex"), encoding="utf-8").read()
allsrc = "\n".join(srcs.values()) + "\n" + supp
L.append("\n## Phase 3B form checks\n")
stale = [r"\(C[1-4]\)", "Evidence level", "evidence level", "systematic literature search", "verify\\_r23\\_numbers", "verify_r23_numbers", "Theorem~R0", "Thm R0", "Round~0 and Theorem R0"]
for pat in stale:
    n = len(re.findall(pat, re.sub(r"(?<!\\)%[^\n]*", "", srcs["60_results.tex"] if pat.startswith("\\(C") else allsrc)))
    L.append(f"- stale/removed phrase `{pat}` in sources: {n}\n"); ok &= n == 0
hl = re.findall(r"\\item (.*)", srcs["10_front.tex"][srcs["10_front.tex"].index("begin{highlights}"):srcs["10_front.tex"].index("end{highlights}")])
hf = [l.strip() for l in open(os.path.join(R, "docs", "HIGHLIGHTS.txt"), encoding="utf-8") if l.strip()]
L.append(f"- Highlights: {len(hl)} items, max length {max(map(len, hl))} (limit 85), HIGHLIGHTS.txt identical to manuscript: {hl == hf}\n"); ok &= len(hl) <= 5 and max(map(len, hl)) <= 85 and hl == hf
bib = srcs["99_bib.tex"]; keys = re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", bib)
body = re.sub(r"(?<!\\)%[^\n]*", "", "\n".join(v for k, v in srcs.items() if k != "99_bib.tex") + supp)
cited = set(k.strip() for g in re.findall(r"\\[cC]ite[a-z]*\{([^}]+)\}", body) for k in g.split(","))
L.append(f"- bibliography: {len(keys)} items; uncited: {sorted(set(keys) - cited)}; cited but missing: {sorted(cited - set(keys))}\n"); ok &= not (set(keys) - cited) and not (cited - set(keys))
xs = open(os.path.join(R, "xref_S.tex"), encoding="utf-8").read(); xm = open(os.path.join(R, "xref_M.tex"), encoding="utf-8").read()
uS = set(re.findall(r"\\xS\{([^}]+)\}", re.sub(r"(?<!\\)%[^\n]*", "", "\n".join(srcs.values())))); uM = set(re.findall(r"\\xM\{([^}]+)\}", re.sub(r"(?<!\\)%[^\n]*", "", supp)))
mS = sorted(l for l in uS if "xS@%s\\endcsname" % l not in xs); mM = sorted(l for l in uM if "xM@%s\\endcsname" % l not in xm)
L.append(f"- cross-document labels: {len(uS)} \\xS and {len(uM)} \\xM used; unresolved: {mS + mM}\n"); ok &= not (mS + mM)

saux = open(os.path.join(R, "CMPB_REVISED_SUPPLEMENT.aux"), encoding="utf-8", errors="replace").read()
slabs = {m.group(1): m.group(2) for m in re.finditer(r"\\newlabel\{((?:fig|tab):[^}@]+)\}\{\{([^}]*)\}", saux)}
nocom = lambda t: re.sub(r"(?<!\\)%[^\n]*", "", t)
refsup = set(l for g in re.findall(r"\\(?:[cC]ref|ref|autoref)\{([^}]*)\}", nocom(supp)) for l in g.split(","))
refmain = set(re.findall(r"\\xS\{([^}]+)\}", nocom("\n".join(srcs.values()))))
unref = sorted(slabs[l] for l in slabs if l not in refsup and l not in refmain)
L.append(f"- supplement floats ({len(slabs)} labelled figures/tables) never cited in the supplement or the main text: {unref}\n"); ok &= not unref
for n in ("CMPB_REVISED_MAIN", "CMPB_REVISED_SUPPLEMENT"):
    lg = open(os.path.join(R, n + ".log"), encoding="utf-8", errors="replace").read()
    ov, un = lg.count("Overfull"), len(re.findall(r"undefined", lg))
    L.append(f"- {n}.log: Overfull {ov}, 'undefined' {un}; PDF pages {subprocess.run(['pdfinfo', os.path.join(R, n + '.pdf')], capture_output=True, text=True).stdout.split('Pages:')[1].split()[0]}\n"); ok &= ov == 0 and un == 0
L.append(f"\n**Overall (automatic part): {'PASS' if ok else 'CHECK'}**\n")
open(out, "w", encoding="utf-8").write("".join(L)); print("written", out, "PASS" if ok else "CHECK")
