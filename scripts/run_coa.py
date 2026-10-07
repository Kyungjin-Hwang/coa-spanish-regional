# Spanish_variable/run_coa.py
from __future__ import annotations
import argparse, json, sys, csv
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from coa_agents import Planner, Retriever, Generator, Auditor
from dialect_runtime import load_rules, normalize_rules
from tag_lexicon import load_lexicon, build_index, add_rules_to_index  # ← rules 보강용


def dummy_generate_fn(prompt: str, n: int) -> List[str]:
    base = " ".join(prompt.split()[-40:])
    return [f"[CAND {i+1}] {base}" for i in range(n)]

def load_generate_fn(temperature: float, max_tokens: int):
    try:
        from my_llm_client import call_model
        def wrapped(prompt: str, n: int) -> List[str]:
            return call_model(prompt, n=n, temperature=temperature, max_tokens=max_tokens)
        print("[INFO] Using my_llm_client.call_model()", file=sys.stderr)
        return wrapped
    except Exception as e:
        print(f"[WARN] my_llm_client not found or failed to import ({e}). Using dummy generator.", file=sys.stderr)
        return lambda prompt, n: dummy_generate_fn(prompt, n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", default="lexicons_out/lexicon_combined.json")
    ap.add_argument("--rules",   default="lexicons_out/rules.yml")

    # ── Retriever 코퍼스 옵션 ───────────────────────────────────────────
    ap.add_argument("--corpus",  default="data/region_corpus.jsonl",
                    help="기본 지역 코퍼스(JSONL, {region,text,source})")
    ap.add_argument("--corpus_phrases", default=None,
                    help="숙어/관용구 전용 코퍼스(JSONL). 지정 시 예문을 섞어 사용")
    ap.add_argument("--phr_mix", type=float, default=0.3,
                    help="리트리버가 숙어 코퍼스를 섞는 비율(0.0~1.0). 예: 0.3이면 30%%")

    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)

    # Planner/Retriever/Generator 하이퍼파라미터
    ap.add_argument("--k_keywords", type=int, default=5)
    ap.add_argument("--k_examples", type=int, default=3)
    ap.add_argument("--n_cands",   type=int, default=6)

    # Auditor 가중치
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--beta",  type=float, default=0.25)
    ap.add_argument("--gamma", type=float, default=0.20)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--epsilon", type=float, default=0.15, help="soft_ban_ratio 감점 가중치")

    # Generator(LM) 파라미터
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max_tokens",  type=int,   default=256)

    # 하드 제약(CoA-H)
    ap.add_argument("--apply_hard", action="store_true")

    # 기본 사용자 언어(입력에 없을 때 사용)
    ap.add_argument("--default_user_language", default="ko")

    # 디버그 TSV 로깅 (옵션)
    ap.add_argument("--debug_tsv", type=str, default=None,
                    help="각 후보별 피처/점수를 TSV로 append 저장(열: run_id, id, target_region, cand_idx, is_best, score, quality, dialect_prob, lexicon_hit_ratio, banned_hits, banned_hit_ratio, soft_ban_ratio, total_matches)")

    args = ap.parse_args()

    in_path  = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)
    if not in_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 리소스 로딩 ───────────────────────────────────────
    lexicon = load_lexicon(args.lexicon)
    index   = build_index(lexicon)
    rules   = normalize_rules(load_rules(args.rules))
    # rules의 prefer 어휘를 인덱스에 보강(사전에 없어도 매칭되게)
    add_rules_to_index(index, rules)

    planner = Planner(lexicon, rules)

    # ── Retriever 초기화: 보조 코퍼스 + 믹스 비율 지원 ──────────────────
    retriever: Optional[Retriever] = None
    if args.k_examples > 0:
        try:
            retriever = Retriever(
                corpus_path=args.corpus,
                corpus_phrases_path=args.corpus_phrases,
                mix_ratio_phrases=max(0.0, min(1.0, float(args.phr_mix))),
            )
        except TypeError:
            # 만약 기존 Retriever 시그니처(보조 코퍼스 미지원)일 경우의 하위호환
            retriever = Retriever(args.corpus)

    generate_fn = load_generate_fn(args.temperature, args.max_tokens)
    generator   = Generator(generate_fn)
    auditor     = Auditor(
        lexicon, index, rules,
        weights={"alpha": args.alpha, "beta": args.beta, "gamma": args.gamma, "delta": args.delta, "epsilon": args.epsilon}
    )

    # ── helper: TSV writer ────────────────────────────────
    def _append_debug_rows(tsv_path: str | None, rows: list[dict], run_id: str):
        if not tsv_path:
            return
        path = Path(tsv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = (not path.exists())
        fieldnames = [
            "run_id","id","target_region","cand_idx","is_best",
            "score","quality","dialect_prob","lexicon_hit_ratio",
            "banned_hits","banned_hit_ratio","soft_ban_ratio","total_matches"
        ]
        with path.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            if write_header:
                w.writeheader()
            for r in rows:
                for k in fieldnames:
                    r.setdefault(k, "")
                r["run_id"] = run_id
                w.writerow(r)

    wrote = 0
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    with in_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            ex = json.loads(line)

            # 안전한 입력 필드 폴백
            source_text  = ex.get("input_text") or ex.get("text") or ex.get("src")
            if not source_text:
                raise ValueError(f"missing input text in example: {ex.get('id')}")

            target_region = ex.get("target_region") or ex.get("region_target") or "ES"
            user_lang     = ex.get("user_language", args.default_user_language)
            style         = ex.get("style", "neutral")
            task_desc     = ex.get("task_desc", ex.get("task", "Translate/Rewrite"))

            # 1) Planner
            policy = planner.plan(
                user_language=user_lang,
                target_region=target_region,
                style=style,
                k_keywords=args.k_keywords,
                register=ex.get("register")
            )

            # 2) Retriever
            examples = []
            if retriever is not None and args.k_examples > 0:
                examples = retriever.retrieve(policy, k_examples=args.k_examples)

            # 3) Generator
            n = int(ex.get("n", args.n_cands))
            cands = generator.generate(
                policy, examples,
                task_desc,
                source_text,
                n=n
            )

            # 4) Auditor (rerank + optional hard constraint)
            qualities = ex.get("qualities")
            if qualities is None:
                try:
                    from quality_heuristics import quality_score
                    qualities = [float(quality_score(c)) for c in cands]
                except Exception as e:
                    qualities = [0.0] * len(cands)
                    print(f"[WARN] quality_heuristics not available ({e}); using 0.0", file=sys.stderr)

            best_idx, scored = auditor.rerank(
                cands, policy["target_region"], qualities=qualities
            )
            best_text = cands[best_idx] if best_idx >= 0 and cands else ""

            # Debug TSV rows
            debug_rows = []
            for (i, s, feats) in scored:
                debug_rows.append({
                    "id": ex.get("id"),
                    "target_region": policy["target_region"],
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
                })
            _append_debug_rows(args.debug_tsv, debug_rows, run_id)

            hard_applied = False
            hard_repls = []
            if args.apply_hard and best_text:
                new_text, changed, repls = auditor.hard_constrain(best_text, policy["target_region"])
                if changed:
                    best_text = new_text
                    hard_applied = True
                    hard_repls = repls

            out = {
                "id": ex.get("id"),
                "policy": policy,
                "examples": examples,
                "best_index": best_idx,
                "best_text": best_text,
                "scores": [{"idx": i, "score": float(s), **feats} for (i, s, feats) in scored],
                "hard_applied": hard_applied,
                "hard_replacements": hard_repls
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            wrote += 1

    print(f"[OK] wrote {wrote} lines → {out_path}")

if __name__ == "__main__":
    main()
