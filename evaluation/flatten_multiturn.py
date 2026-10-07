#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flatten_multiturn.py
Convert nested multiturn_outputs.jsonl to flat format for score_multiturn_consistency.py

Input format (nested):
  {"id": "mt_0001", "region": "ES", "systems": {"B1": {"outputs": [...]}, "CoA_S": {"outputs": [...]}, ...}}

Output format (flat, one record per session per system):
  {"id": "mt_0001", "region": "ES", "system": "B1", "turns": [...]}
  {"id": "mt_0001", "region": "ES", "system": "CoA_S", "turns": [...]}
"""
import json, argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="out/multiturn_outputs.jsonl")
    ap.add_argument("--output", default="out/multiturn_outputs_flat.jsonl")
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.input, encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = rec.get("id") or rec.get("session_id")
            region = rec.get("region", "ES")
            seed_lexemes = rec.get("seed_lexemes", [])
            systems = rec.get("systems", {})

            for sysname, sysdata in systems.items():
                if "error" in sysdata:
                    continue
                flat = {
                    "id": sid,
                    "session_id": sid,
                    "region": region,
                    "target_region": region,
                    "system": sysname,
                    "system_name": sysname,
                    "seed_lexemes": seed_lexemes,
                    "turns": sysdata.get("outputs", []),
                    "turn_texts": sysdata.get("outputs", []),
                    "outputs": sysdata.get("outputs", []),
                    "responses": sysdata.get("outputs", []),
                }
                fout.write(json.dumps(flat, ensure_ascii=False) + "\n")
                n += 1

    print(f"[OK] Wrote {n} records → {args.output}")

if __name__ == "__main__":
    main()
