# Spanish_variable/runs/run_baseline_b1.py

from __future__ import annotations
import argparse, json, sys, csv, os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# ==== [핵심 수정] 루트 경로를 sys.path에 추가 ====
# 이 파일은 Spanish_variable/runs/ 안에 있다고 가정
_THIS_FILE = Path(__file__).resolve()
ROOT = _THIS_FILE.parent.parent        # Spanish_variable/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 위치별 fallback import (루트 직하 or 하위패키지에 있을 때 모두 커버)
try:
    from dialect_runtime import (
        load_rules, normalize_rules,
        auditor_score, rerank_candidates, classify_region_rule
    )
except ModuleNotFoundError:
    # 예: Spanish_variable/runtime/dialect_runtime.py 같은 구조인 경우
    try:
        from runtime.dialect_runtime import (
            load_rules, normalize_rules,
            auditor_score, rerank_candidates, classify_region_rule
        )
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Cannot import 'dialect_runtime'. "
            f"Checked ROOT={ROOT} and subpackages (e.g., runtime/). "
            "Confirm the file exists (dialect_runtime.py) and adjust import paths."
        ) from e

try:
    from tag_lexicon import load_lexicon, build_index, add_rules_to_index
except ModuleNotFoundError:
    try:
        from runtime.tag_lexicon import load_lexicon, build_index, add_rules_to_index
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Cannot import 'tag_lexicon'. "
            f"Checked ROOT={ROOT} and subpackages (e.g., runtime/). "
            "Confirm the file exists (tag_lexicon.py) and adjust import paths."
        ) from e
# ==== [핵심 수정 끝] ====


def _append_debug_rows(tsv_path: Optional[str], rows: List[Dict[str, Any]]):
    if not tsv_path:
        return
    path = Path(tsv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not path.exists())
    fieldnames = [
        "id","target_region","cand_idx","is_best",
        "score","quality","dialect_prob","lexicon_hit_ratio",
        "banned_hits","banned_hit_ratio","soft_ban_ratio","total_matches","pass_label"
    ]
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        if write_header:
            w.writeheader()
        for r in rows:
            for k in fieldnames:
                r.setdefault(k, "")
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description="Baseline B1: 사전 hit 기반 평가/라벨러")
    ap.add_argument("--lexicon", default="lexicons_out/lexicon_combined.json")
    ap.add_argument("--rules",   default="lexicons_out/rules.yml")
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)

    # 스코어 가중치
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--beta",  type=float, default=0.25)
    ap.add_argument("--gamma", type=float, default=0.20)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--epsilon", type=float, default=0.15, help="soft_ban_ratio 감점")

    # 라벨 임계치(간단 합/격리 리포트용)
    ap.add_argument("--tau", type=float, default=0.5, help="score >= tau → PASS")

    # 디버그 TSV
    ap.add_argument("--debug_tsv", type=str, default=None,
                    help="각 예제/후보의 피처/점수 TSV 로그를 append로 저장")

    args = ap.parse_args()

    in_path  = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)
    if not in_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 리소스 로딩
    lexicon = load_lexicon(args.lexicon)
    index   = build_index(lexicon)
    rules   = normalize_rules(load_rules(args.rules))
    add_rules_to_index(index, rules)  # prefer 보강

    weights = {"alpha": args.alpha, "beta": args.beta, "gamma": args.gamma, "delta": args.delta, "epsilon": args.epsilon}

    wrote = 0
    with in_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            ex = json.loads(line)

            target = ex.get("target_region")
            if not target:
                raise ValueError(f"missing target_region in: {ex}")

            # 입력 형태 케이스들:
            #  A) 단일 텍스트 평가: ex['text'] 또는 ex['input_text']
            #  B) 후보 리스트 평가: ex['candidates'] (+ qualities 선택)
            if "candidates" in ex and isinstance(ex["candidates"], list):
                cands: List[str] = ex["candidates"]
                quals: Optional[List[float]] = ex.get("qualities")
                # 재랭크
                best_idx, scored = rerank_candidates(
                    cands, target, index, rules, qualities=quals, weights=weights
                )
                best_text = cands[best_idx] if best_idx >= 0 and cands else ""

                # PASS 라벨은 best 기준으로만 판단
                best_score = next((s for (i, s, _f) in scored if i == best_idx), None)
                pass_label = (best_score is not None and best_score >= args.tau)

                # 디버그 TSV
                debug_rows = []
                for (i, s, feats) in scored:
                    debug_rows.append({
                        "id": ex.get("id"),
                        "target_region": target,
                        "cand_idx": i,
                        "is_best": int(i == best_idx),
                        "score": float(s),
                        "quality": feats.get("quality", 0.0),
                        "dialect_prob": feats.get("dialect_prob", 0.0),
                        "lexicon_hit_ratio": feats.get("lexicon_hit_ratio", 0.0),
                        "banned_hits": feats.get("banned_hits", 0),
                        "banned_hit_ratio": feats.get("banned_hit_ratio", 0.0),
                        "soft_ban_ratio": feats.get("soft_ban_ratio", 0.0),
                        "total_matches": feats.get("total_matches", 0),
                        "pass_label": int(pass_label) if i == best_idx else "",
                    })
                _append_debug_rows(args.debug_tsv, debug_rows)

                # 출력 레코드
                out = {
                    "id": ex.get("id"),
                    "mode": "candidates",
                    "target_region": target,
                    "best_index": best_idx,
                    "best_text": best_text,
                    "scores": [{"idx": i, "score": float(s), **feats} for (i, s, feats) in scored],
                    "pass": bool(pass_label),
                    "tau": float(args.tau),
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                wrote += 1
                continue

            # 단일 텍스트 평가
            text = ex.get("text") or ex.get("input_text")
            if not text:
                raise ValueError(f"missing 'text'/'input_text' or 'candidates' in: {ex}")

            # 점수/피처
            score, feats, _res = auditor_score(
                text, target, index, rules,
                quality_score=float(ex.get("quality", 0.0)),
                weights=weights
            )
            pass_label = (score >= args.tau)

            # 사전 기반 지역 분류 예측(참고용)
            pred_region, _ = classify_region_rule(text, index)

            # 디버그 TSV
            _append_debug_rows(args.debug_tsv, [{
                "id": ex.get("id"),
                "target_region": target,
                "cand_idx": 0,
                "is_best": 1,
                "score": float(score),
                "quality": feats.get("quality", 0.0),
                "dialect_prob": feats.get("dialect_prob", 0.0),
                "lexicon_hit_ratio": feats.get("lexicon_hit_ratio", 0.0),
                "banned_hits": feats.get("banned_hits", 0),
                "banned_hit_ratio": feats.get("banned_hit_ratio", 0.0),
                "soft_ban_ratio": feats.get("soft_ban_ratio", 0.0),
                "total_matches": feats.get("total_matches", 0),
                "pass_label": int(pass_label),
            }])

            out = {
                "id": ex.get("id"),
                "mode": "single",
                "target_region": target,
                "text": text,
                "score": float(score),
                "features": feats,
                "pass": bool(pass_label),
                "tau": float(args.tau),
                "pred_region_hit": pred_region,  # 참고용 분류기 결과
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            wrote += 1

    print(f"[OK] wrote {wrote} lines → {out_path}")


if __name__ == "__main__":
    main()
