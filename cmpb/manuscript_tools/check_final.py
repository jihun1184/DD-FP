#!/usr/bin/env python3
"""Check the FINAL main text and supplement (this folder) against the stored records in data/.
1. recompute headline numbers from data/ and require that they appear in the compiled PDF text;
2. require absence of unresolved references and of '??';
3. require scope phrases that must stay in the text.
Usage: python3 tools/check_final.py   (after tools/assemble.sh). Exit code 1 on any failure."""
import csv, json, os, re, statistics as st, subprocess, sys
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
def txt(f):
    path = os.path.join(R, f)
    import shutil
    extractor = shutil.which("pdftotext")
    if extractor:
        try:
            result = subprocess.run([extractor, path, "-"], capture_output=True, text=True, encoding="utf-8", check=True)
            return re.sub(r"\s+", "", result.stdout)
        except (OSError, subprocess.CalledProcessError):
            pass
    from pypdf import PdfReader
    text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return re.sub(r"\s+", "", text)
main, supp = txt("CMPB_REVISED_MAIN.pdf"), txt("CMPB_REVISED_SUPPLEMENT.pdf")
res = []
def chk(name, ok): res.append((name, bool(ok)))
def has(t, *needles): return all(n in t for n in needles)
D = os.path.join(R, "data")
# --- G5 (tier G) from stored per-volume rows
sub = list(csv.DictReader(open(os.path.join(D, "subject_table.csv"))))
m = lambda k: st.median(float(r[k]) for r in sub)
g = {"frozen/AF": m("speedup_af"), "frozen/pile": m("speedup_pile"), "AF/pile": m("r_af_over_pile")}
chk("G5 n=20 volumes", len(sub) == 20)
chk("G5 pile check OK 20/20", all(r["pile_check"] == "OK" for r in sub))
for k, v in g.items(): chk(f"G5 median {k} = {v:.2f} in main", has(main, f"{v:.2f}"))
for k in ("t_frozen_s", "t_af_s", "t_pile_s"): chk(f"G5 median {k} = {m(k):.2f} in main", has(main, f"{m(k):.2f}"))
# --- memory (tier M)
mem = json.load(open(os.path.join(D, "p2_memory_recompute.json")))
chk("memory 59 records", sum(mem["n_records"].values()) == 59)
rat = [x["obs_over_model"] for x in mem["model_vs_obs"]]
chk("memory ledger ratios within 1.0000004-1.0000117", min(rat) > 1.0000003 and max(rat) < 1.0000118)
agg = mem["aggregate"]; S3 = {(r["m"], r["method"], r["K"]): r for r in agg if r["src"] == "S3"}
for cfg, lo_hi in (("B", (0.155, 0.310)), ("C", (0.712, 0.779))):
    v = [r["live_MiB_med"] / S3[(r["m"], "A", 1)]["live_MiB_med"] for (mm, me, K), r in S3.items() if me == cfg]
    chk(f"memory {cfg}/A (S3 rows) within {lo_hi}", lo_hi[0] - 0.002 <= min(v) and max(v) <= lo_hi[1] + 0.002)
chk("memory ranges in main", has(main, "0.165", "0.233", "0.155", "0.310", "0.717", "0.742"))
# --- tier S
cells = list(csv.DictReader(open(os.path.join(D, "tierS_time_ratios_from_frozen_cli.csv"))))
b = [float(c["B_over_A"]) for c in cells]; c_ = [float(c["C_over_A"]) for c in cells]; cb = [float(c["C_over_A"]) / float(c["B_over_A"]) for c in cells]
chk("tier S cells = 15", len(cells) == 15)
chk("tier S C/B 2.89-3.06", abs(min(cb) - 2.89) < 0.01 and abs(max(cb) - 3.06) < 0.01)
chk("tier S ratios in main", has(main, "0.32", "0.52", "2.89", "3.06", "0.94", "1.49"))
# --- fixed headline numbers carried from records (presence only)
for n in ["282,015,220", "8.05", "7.62", "8.51", "9,544", "451.9", "10,008", "1,251", "1,689", "5,696", "26.78", "649,707,520"]:
    chk(f"carried number {n} in main", has(main, n) or has(main, n.replace(",", "")))
for n in ["0.8362", "0.8759", "12.59", "281"]:
    chk(f"E1 number {n} in supplement", has(supp, n))
# --- scope phrases that must stay
chk("independent checker limited to two arrays", has(main, "twosaveddirectassemblyarrays") or has(main, "twodirectassemblyarrays"))
chk("proof status (not machine-verified, not externally reviewed) stated", has(main, "machine-verified") and has(main, "reviewedexternally"))
chk("H-GPU unproved stated", has(main, "H-GPU"))
chk("prior-result table present", has(main, "Lemma5", "Theorem2", "KeyLemma"))
chk("Mazo cited", has(main, "Mazo"))
chk("negative reconstruction outcome stated", has(main, "didnotexceedbothlocaloperators") or has(main, "didnotfavorFP"))
chk("no unresolved '??'", "??" not in main and "??" not in supp)
for d in ("CMPB_REVISED_MAIN", "CMPB_REVISED_SUPPLEMENT"):
    lg = open(os.path.join(R, d + ".log"), errors="replace").read()
    chk(f"{d}: no undefined refs/citations", not re.search(r"undefined", lg))
# --- R0 re-verification record (supplement S-A.7) vs stored result files
V = os.path.join(R, "verify_r0")
rs = [json.load(open(os.path.join(V, f"results_random_{x}.json"))) for x in ("20261027", "20261028", "20261029", "20261104")]
chk("R0 random: windows runs sum 71,685", sum(r["windows_runs"] for r in rs) == 71685 and has(supp, "71,685"))
chk("R0 random: pairs sum 175,759,600", sum(r["pairs"] for r in rs) == 175759600)
chk("R0 random: 22,000 volumes", sum(r["volumes"] for r in rs) == 22000 and has(supp, "22,000"))
chk("R0 random: all violation counters zero", all(v == 0 for r in rs for k, v in r.items() if k not in ("volumes", "windows_runs", "pairs", "by_kind", "_meta")))
ar = json.load(open(os.path.join(V, "results_arbitrary_20261101.json")))
chk("R0 arbitrary: 72,000 runs, 30,055,380 pairs, 4 maps distinct", ar["runs"] == 72000 and ar["pairs"] == 30055380 and ar["maps_with_distinct_outputs"] == 4 and has(supp, "30,055,380"))
chk("R0 arbitrary: zero violations", all(ar[k] == 0 for k in ("adm_bad", "range", "F1", "F2", "F3", "KL", "P", "Q")))
mu = json.load(open(os.path.join(V, "results_mutants_20261030.json")))
chk("R0 mutants: P3 violator KL 22,853 / F3 131,682 / B_dwc 165 vols", mu["M1_P3_violating_scheduler"]["KL"] == 22853 and mu["M1_P3_violating_scheduler"]["F3"] == 131682 and mu["M1_P3_violating_scheduler"]["vols_with_B_dwc_violation"] == 165 and has(supp, "22,853", "131,682"))
chk("R0 mutants: altered hypotheses never create DWC violation", all(mu[k]["B_dwc"] == 0 for k in ("M2_frame_alpha1_start1", "M3_halo_delta2", "M4_descending_owner_at_seam")))
chk("R0 baseline zero", all(v == 0 for k, v in mu["baseline"].items() if k != "volumes"))
chk("R0 sampled (not exhaustive) wording kept", has(supp, "sampled,not enumerated") or has(supp, "sampled,notenumerated"))
# --- stale / forbidden wording
for bad_phrase in ("one in-project review round", "design-ai", "O(N/K)\\text{ saving}", "Domain-Decomposed GPU Front Propagation"):
    chk(f"absent: {bad_phrase}", not has(main.replace(" ", ""), bad_phrase.replace(" ", "")))
chk("title (new) in main", has(main, "DD-FP:GPUFrontPropagationwithSlabDomainDecomposition"))
chk("7.48 separate baseline statement", has(main, "7.48") and (has(main, "differentbaselines") or has(main, "separatebaseline")))
# --- no stale files
chk("no stale segment 70_experiments", not os.path.exists(os.path.join(R, "seg", "70_experiments.tex")))
bad = [n for n, ok in res if not ok]
for n, ok in res: print(("PASS " if ok else "FAIL ") + n)
print(f"\n{len(res)-len(bad)}/{len(res)} passed"); sys.exit(1 if bad else 0)
