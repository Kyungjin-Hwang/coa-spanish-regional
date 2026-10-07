#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_multiturn_consistency.py
Directly compute Session Target Share Mean from multiturn_outputs.jsonl
and multiturn_sessions.jsonl, handling the nested systems format.

Usage:
  python eval_multiturn_consistency.py \
    --sessions data/multiturn_sessions.jsonl \
    --outputs out/multiturn_outputs.jsonl \
    --out_csv reports/consistency_50sessions.csv
"""
import json, argparse
from pathlib import Path
from collections import defaultdict

def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def compute_target_share(turns, seed_lexemes):
    """Compute per-turn target lexeme hit rate, then average across turns."""
    if not turns or not seed_lexemes:
        return 0.0
    per_turn = []
    for t in turns:
        t_lower = t.lower()
        hits = sum(1 for lex in seed_lexemes if lex.lower() in t_lower)
        per_turn.append(min(hits, 1))  # binary: did any lexeme appear?
    return sum(per_turn) / len(per_turn)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="data/multiturn_sessions.jsonl")
    ap.add_argument("--outputs", default="out/multiturn_outputs.jsonl")
    ap.add_argument("--out_csv", default="reports/consistency_50sessions.csv")
    args = ap.parse_args()

    sessions = load_jsonl(args.sessions)
    outputs = load_jsonl(args.outputs)

    # Build session metadata
    meta = {}
    for s in sessions:
        sid = s.get("id") or s.get("session_id")
        meta[sid] = {
            "region": s.get("region", "ES"),
            "seed_lexemes": s.get("seed_lexemes", [])
        }

    # Collect results
    results = defaultdict(lambda: defaultdict(list))

    for rec in outputs:
        sid = rec.get("id") or rec.get("session_id")
        region = rec.get("region") or (meta.get(sid, {}).get("region", "ES"))
        seed_lexemes = rec.get("seed_lexemes") or (meta.get(sid, {}).get("seed_lexemes", []))
        systems = rec.get("systems", {})

        for sysname, sysdata in systems.items():
            if "error" in sysdata:
                continue
            turns = sysdata.get("outputs", [])
            share = compute_target_share(turns, seed_lexemes)
            results[(sysname, region)]["shares"].append(share)

    # Aggregate and write CSV
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for (sysname, region), data in sorted(results.items()):
        shares = data["shares"]
        mean_share = sum(shares) / len(shares) if shares else 0.0
        rows.append({
            "system": sysname,
            "region": region,
            "session_target_share_mean": round(mean_share, 2),
            "n_sessions": len(shares)
        })

    # Print table
    print(f"{'Region':<8} {'System':<10} {'Target Share Mean':>20} {'n_sessions':>12}")
    print("-" * 55)
    for r in sorted(rows, key=lambda x: (x["region"], x["system"])):
        print(f"{r['region']:<8} {r['system']:<10} {r['session_target_share_mean']:>20.2f} {r['n_sessions']:>12}")

    # Write CSV
    import csv
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["region", "system", "session_target_share_mean", "n_sessions"])
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["region"], x["system"])):
            w.writerow(r)

    print(f"\n[OK] Wrote {args.out_csv}")

if __name__ == "__main__":
    main()
