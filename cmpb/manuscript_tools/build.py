"""Build the main manuscript and supplement from a single root.
Usage: python tools/build.py [--tex /path/to/pdflatex] [--regenerate-figures]
"""
from pathlib import Path
import argparse, re, subprocess, sys
root=Path(__file__).resolve().parent.parent
parser=argparse.ArgumentParser()
parser.add_argument("--tex", default="pdflatex")
parser.add_argument("--regenerate-figures", action="store_true")
args=parser.parse_args()
parts="00_preamble 10_front 20_intro 30_background 40_method 50_theory 60_results 70_discussion 95_declarations 99_bib".split()
(root/"CMPB_REVISED_MAIN.tex").write_text("".join((root/"seg"/(x+".tex")).read_text(encoding="utf-8")+"\n" for x in parts),encoding="utf-8")
if args.regenerate_figures:
    subprocess.run([sys.executable,"tools/make_figs.py"],cwd=root,check=True)
subprocess.run([sys.executable,"tools/make_tables.py"],cwd=root,check=True,capture_output=True)
for cycle in range(4):
    for name in ["CMPB_REVISED_MAIN","CMPB_REVISED_SUPPLEMENT"]:
        result=subprocess.run([args.tex,"-interaction=nonstopmode","-halt-on-error",name+".tex"],cwd=root,capture_output=True)
        (root/("build_"+name+".log")).write_bytes(result.stdout+result.stderr)
        if result.returncode:
            print(result.stdout.decode("utf-8",errors="replace")[-2500:])
            raise SystemExit(result.returncode)
    subprocess.run([sys.executable,"tools/make_xref.py","."],cwd=root,check=True,capture_output=True)
for name in ["CMPB_REVISED_MAIN","CMPB_REVISED_SUPPLEMENT"]:
    log=(root/(name+".log")).read_text(encoding="utf-8",errors="replace")
    if re.search(r"undefined|^!",log,re.M):
        raise SystemExit("Unresolved reference, citation or LaTeX error in "+name)
subprocess.run([sys.executable,"tools/check_final.py"],cwd=root,check=True)
print("Main manuscript and supplement built; final checks passed.")
