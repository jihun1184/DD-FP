#!/usr/bin/env python3
"""Pre-registered cost gate: project full-tier wall time from a pilot summary; exit 4 (NEEDS_APPROVAL) if over budget."""
import json, sys
pilot, tier, budget_min = sys.argv[1], sys.argv[2], float(sys.argv[3])
b = json.load(open(sys.argv[4]))[tier]["cases"]
s = json.load(open(f"{pilot}/{tier}_summary.json"))
if s["completed"] == 0 or s["verdict"] not in ("PASS", "CONDITIONAL", "INCOMPLETE"): print("pilot unusable:", s["verdict"]); sys.exit(3)
proj = s["wall_s"] / s["completed"] * b / 60
print(json.dumps(dict(tier=tier, pilot_cases=s["completed"], pilot_s=s["wall_s"], projected_full_min=round(proj, 1), budget_min=budget_min,
                      peak_rss_mb=round(s["peak_rss_mb"], 1))))
sys.exit(4 if proj > budget_min else 0)
