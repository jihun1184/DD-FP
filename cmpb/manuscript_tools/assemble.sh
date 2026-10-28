#!/bin/bash
# Build main text and supplement from the segments in seg/. Fails (non-zero exit) on any LaTeX error,
# undefined reference/citation, or missing PDF. Run from anywhere:  bash tools/assemble.sh
# Optional: REGEN_FIGS=1 regenerates the run-time and memory figures from data/ (needs matplotlib).
set -euo pipefail
cd "$(dirname "$0")/.."
SEG="00_preamble 10_front 20_intro 30_background 40_method 50_theory 60_results 70_discussion 95_declarations 99_bib"
: > CMPB_REVISED_MAIN.tex
for s in $SEG; do cat "seg/$s.tex" >> CMPB_REVISED_MAIN.tex; echo >> CMPB_REVISED_MAIN.tex; done
[ "${REGEN_FIGS:-0}" = "1" ] && python3 tools/make_figs.py
python3 tools/make_tables.py > /dev/null
tex() { pdflatex -interaction=nonstopmode -halt-on-error "$1.tex" > "build_$1.log" 2>&1 || { echo "LaTeX failed: $1 (see build_$1.log)"; tail -20 "build_$1.log"; exit 1; }; }
for i in 1 2 3; do tex CMPB_REVISED_MAIN; tex CMPB_REVISED_SUPPLEMENT; python3 tools/make_xref.py . > /dev/null; done
tex CMPB_REVISED_MAIN; tex CMPB_REVISED_SUPPLEMENT
rc=0
for d in CMPB_REVISED_MAIN CMPB_REVISED_SUPPLEMENT; do
  n=$(grep -c -E "undefined|Citation .* undefined|Reference .* undefined" $d.log || true)
  echo "$d: pages=$(pdfinfo $d.pdf | awk '/Pages/{print $2}') undefined=$n"
  [ "$n" != "0" ] && rc=1
done
exit $rc
