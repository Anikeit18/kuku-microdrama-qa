#!/usr/bin/env python3
"""Score QA reports against samples/truth.json.

A fault counts as caught if a non-info issue of the same check overlaps it (+-1 s).
Every other non-info issue is a false alarm.

    python tests/evaluate.py --samples samples --reports outputs/with_script [--label with_script]
"""
import argparse
import json
import os
from collections import defaultdict


def overlaps(i, t, tol=1.0):
    return i["start"] <= t["end"] + tol and i["end"] >= t["start"] - tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="samples")
    ap.add_argument("--reports", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    truth = json.load(open(os.path.join(a.samples, "truth.json"), encoding="utf-8"))
    nl = os.path.join(a.samples, "natural_labels.json")
    if os.path.exists(nl):  # faults a human saw while watching (real test set)
        for k, v in json.load(open(nl, encoding="utf-8")).items():
            if k in truth and isinstance(v, list):
                truth[k] += [dict(x, source="natural") for x in v]
    per_check = defaultdict(lambda: [0, 0, 0])  # caught (any severity), caught (major+), total
    rows, fa_total = [], 0
    for name, faults in truth.items():
        rp = os.path.join(a.reports, name, "report.json")
        if not os.path.exists(rp):
            continue
        rep = json.load(open(rp, encoding="utf-8"))
        flagged = [i for i in rep["issues"] if i["severity"] != "info"]
        used = set()
        caught = []
        for t in faults:
            ok = t["check"] if isinstance(t["check"], list) else [t["check"]]
            key = "/".join(ok)
            hit = [i for i in flagged if i["check"] in ok and overlaps(i, t)]
            per_check[key][2] += 1
            if hit:
                per_check[key][0] += 1
                used.update(i["id"] for i in hit)
            if any(i["severity"] in ("blocker", "major") for i in hit):
                per_check[key][1] += 1
            caught.append(bool(hit))
        false = [i for i in flagged if i["id"] not in used]
        fa_total += len(false)
        rows.append({"variant": name, "verdict": rep["verdict"], "faults": len(faults), "caught": sum(caught),
                     "false_alarms": len(false),
                     "false_alarm_list": [f"{i['check']}@{i['start']}s ({i['severity']})" for i in false]})
    tot = sum(v[2] for v in per_check.values())
    out = {"label": a.label,
           "per_check_recall": {k: {"flagged_any_severity": f"{c}/{n}", "flagged_major_or_blocker": f"{m}/{n}"}
                                for k, (c, m, n) in per_check.items()},
           "faults_caught": f"{sum(v[0] for v in per_check.values())}/{tot}",
           "faults_caught_major_plus": f"{sum(v[1] for v in per_check.values())}/{tot}",
           "false_alarms_total": fa_total, "variants": rows}
    with open(os.path.join(a.reports, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"== {a.label}  faults flagged {out['faults_caught']} (as major/blocker {out['faults_caught_major_plus']}), "
          f"false alarms {fa_total}")
    for k, v in out["per_check_recall"].items():
        print(f"   {k:34s} any {v['flagged_any_severity']:5s} major+ {v['flagged_major_or_blocker']}")
    for r in rows:
        print(f"   {r['variant']:28s} {r['verdict']:7s} caught {r['caught']}/{r['faults']}  FA {r['false_alarms']} {r['false_alarm_list']}")


if __name__ == "__main__":
    main()
