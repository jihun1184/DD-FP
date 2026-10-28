#!/usr/bin/env python3
"""make_xref.py MANUSCRIPT_DIR -- cross-document references without xr-hyper.

xr-hyper + cleveref prints the external file name inside the link text, so the numbers are copied instead:
reads CMPB_REVISED_MAIN.aux and CMPB_REVISED_SUPPLEMENT.aux and writes
  xref_S.tex  (labels of the SUPPLEMENT, used in the main text as \\xS{label})
  xref_M.tex  (labels of the MAIN TEXT, used in the supplement as \\xM{label})
Each macro expands to e.g. 'Figure~S4' or 'Algorithm~1'.  Re-run between LaTeX passes (build_r23.sh does).
"""
import re, sys, os
d = sys.argv[1]
WORD = {"figure": "Figure", "subfigure": "Figure", "table": "Table", "algorithm": "Algorithm", "theorem": "Theorem",
        "lemma": "Lemma", "proposition": "Proposition", "section": "Section", "subsection": "Section",
        "subsubsection": "Section", "equation": "Eq."}
pat = re.compile(r"\\newlabel\{([^}]+)@cref\}\{\{\[([a-z]+)\]\[[^\]]*\]\[[^\]]*\]([^}]*)\}")
def conv(aux, out, pre):
    if not os.path.exists(aux):
        return 0
    lines = []
    for m in pat.finditer(open(aux, encoding="utf-8", errors="replace").read()):
        lab, typ, num = m.groups()
        if typ not in WORD:
            continue
        lines.append("\\expandafter\\gdef\\csname %s@%s\\endcsname{%s~%s}" % (pre, lab, WORD[typ], num))
    new = "\n".join(sorted(set(lines))) + "\n"
    old = open(out, encoding="utf-8").read() if os.path.exists(out) else None
    if old != new:
        open(out, "w", encoding="utf-8").write(new)
    return len(lines)
n1 = conv(os.path.join(d, "CMPB_REVISED_SUPPLEMENT.aux"), os.path.join(d, "xref_S.tex"), "xS")
n2 = conv(os.path.join(d, "CMPB_REVISED_MAIN.aux"), os.path.join(d, "xref_M.tex"), "xM")
print("xref: %d supplement labels -> xref_S.tex, %d main labels -> xref_M.tex" % (n1, n2))
