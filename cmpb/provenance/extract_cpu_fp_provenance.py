#!/usr/bin/env python3
"""Tabulate which cpu_fp.py (SHA-256) is recorded in the case-provenance JSON files of the qualitative topology exports
(BraTS 3 cases, DRIVE 20, CREMI-3D 30).  Reads the archived JSON files; runs nothing else.
Usage: python extract_cpu_fp_provenance.py BRATS_EXPORTS_DIR DRIVE_CREMI_RUN_DIR OUT.csv
  BRATS_EXPORTS_DIR     contains <case>/case_provenance.json          (key r3_active_source_hashes)
  DRIVE_CREMI_RUN_DIR   contains inputs/{DRIVE,CREMI}/cases/<case>/case_provenance.json   (key r3_source_sha256)
Output columns: dataset, case_id, repo_head, cpu_fp_sha256_prefix8, experiment_DDFP_all_prefix8 (BraTS only), seq_fp_time_s
"""
import csv, glob, json, os, sys
brats, run, out = sys.argv[1:4]
rows = []
for f in sorted(glob.glob(os.path.join(brats, "*", "case_provenance.json"))):
    j = json.load(open(f)); h = j["r3_active_source_hashes"]
    rows.append(["BraTS2021", j["case_id"], j["repo_head"][:7], h["src/ddfp/cpu_fp.py"][:8],
                 h.get("scripts/ddfp/experiment_DDFP_all.py", "")[:8], round(j["timing_operational_only_s"]["sequential_fp"], 1)])
for ds in ("DRIVE", "CREMI"):
    for f in sorted(glob.glob(os.path.join(run, "inputs", ds, "cases", "*", "case_provenance.json"))):
        j = json.load(open(f)); h = j["r3_source_sha256"]
        rows.append([ds, j["case_id"], j["r3_git_head"][:7], h["src/ddfp/cpu_fp.py"][:8], "", round(j["timing_seconds"]["seq_fp"], 1)])
with open(out, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["dataset", "case_id", "repo_head", "cpu_fp_sha256_prefix8", "experiment_DDFP_all_prefix8", "seq_fp_time_s"]); w.writerows(rows)
print(len(rows), "cases;", sorted({(r[0], r[3]) for r in rows}))
