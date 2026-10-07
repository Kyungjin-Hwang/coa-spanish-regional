# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, sys, traceback
from pathlib import Path
from typing import List, Dict, Any, Optional

# ---- 안전 임포트 (패키지/스크립트 양쪽 지원) ---
try:
    from .coa_agents import Planner, Retriever, Generator, Auditor
    from .tag_lexicon import load_lexicon, build_index, add_rules_to_index
    from .dialect_runtime import load_rules, normalize_rules
except Exception:
    from coa_agents import Planner, Retriever, Generator, Auditor
    from tag_lexicon import load_lexicon, build_index, add_rules_to_index
    from dialect_runtime import load_rules, normalize_rules

# LLM 호출 래퍼
def load_generate_fn(temperature: float, max_tokens: int):
    try:
        from my_llm_client import call_model
        def wrapped(prompt: str, n: int) -> List[str]:
            return call_model(prompt, n=n, temperature=temperature, max_tokens=max_tokens)
        print("[INFO] Using my_llm_client.call_model()", file=sys.stderr)
        return wrapped
    except Exception as e:
        print(f"[WARN] my_llm_client unavailable ({e}). Using dummy generator.", file=sys.stderr)
        def dummy(prompt: str, n: int) -> List[str]:
            base = " ".join(prompt.split()[-40:])
            return [f"[DUMMY {i+1}] {base}" for i in range(n)]
        return dummy

def build_prompt(system: str, region: str, history: List[str], user: str) -> str:
    """아주 단순한 컨텍스트 프롬프트(시스템별 톤만 바꿈)"""
    sys_tag = {
        "B1":    f"Reescribe/Traduce al español de {region}. Usa léxico propio de la región.",
        "CoA_S": f"CoA-S: Sigue las políticas para {region}.",
        "CoA_R": f"CoA-R: Usa ejemplos relevantes y léxico de {region}.",
        "CoA_H": f"CoA-H: Aplica restricciones duras; evita ban y usa prefer para {region}.",
    }.get(system, f"Léxico objetivo: {region}")
    parts = [f"[SYSTEM] {sys_tag}"]
    for i, turn in enumerate(history):
        role = "USER" if i % 2 == 0 else "ASSISTANT"
        parts.append(f"[{role}] {turn}")
    parts.append(f"[USER] {user}")
    return "\n".join(parts)

def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--systems", nargs="+", default=["B1","CoA_S","CoA_R","CoA_H"])

    # 리소스
    ap.add_argument("--lexicon", default="lexicons_out/lexicon_combined.json")
    ap.add_argument("--rules",   default="lexicons_out/rules.yml")
    ap.add_argument("--corpus",  default="data/region_corpus.jsonl")

    # 생성 파라미터
    ap.add_argument("--n_cands", type=int, default=6)
    ap.add_argument("--k_examples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--apply_hard_for_coa_h", action="store_true")

    # 디버그
    ap.add_argument("--debug_tsv", default=None)

    args = ap.parse_args()

    in_path  = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) 리소스 로딩
    lexicon = load_lexicon(args.lexicon)
    index   = build_index(lexicon)
    rules   = normalize_rules(load_rules(args.rules))
    add_rules_to_index(index, rules)

    # 2) 에이전트 준비
    planner   = Planner(lexicon, rules)
    retriever = Retriever(args.corpus) if args.k_examples > 0 else None
    generate  = load_generate_fn(args.temperature, args.max_tokens)
    generator = Generator(generate)
    auditor   = Auditor(
        lexicon, index, rules,
        weights={"alpha":0.45,"beta":0.25,"gamma":0.20,"delta":0.10,"epsilon":0.15}
    )

    # 3) 입출력 준비
    n_written = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip(): continue
            sess = json.loads(line)
            sid   = sess.get("id")
            region= sess.get("region","ES")
            turns = sess.get("turns",[])
            seed_lex = sess.get("seed_lexemes",[])

            # 시스템별 결과를 누적하여 한 줄로 기록
            rec: Dict[str,Any] = {"id": sid, "region": region, "seed_lexemes": seed_lex, "systems": {}}

            for sysname in args.systems:
                try:
                    history: List[str] = []
                    sys_outs: List[str] = []
                    for t_i, user_utt in enumerate(turns):
                        # 3.1 Planner
                        policy = planner.plan(
                            user_language="es",  # 멀티턴은 이미 ES 맥락
                            target_region=region,
                            style="neutral",
                            k_keywords=5
                        )

                        # 3.2 Retriever
                        examples = []
                        if retriever and ("CoA_" in sysname):
                            examples = retriever.retrieve(policy, k_examples=args.k_examples)

                        # 3.3 Generator
                        prompt = build_prompt(sysname, region, history, user_utt)
                        cands = generator.generate(policy, examples, "Multi-turn continuation", prompt, n=args.n_cands)

                        # 3.4 Auditor (rerank + hard)
                        best_idx, scored = auditor.rerank(cands, region, qualities=None)
                        best_text = cands[best_idx] if (best_idx >= 0 and cands) else ""

                        if args.apply_hard_for_coa_h and sysname == "CoA_H" and best_text:
                            new_text, changed, repls = auditor.hard_constrain(best_text, region)
                            if changed: best_text = new_text

                        sys_outs.append(best_text)
                        # 히스토리에 모델 응답을 추가(다음 턴 컨텍스트로 사용)
                        history.append(user_utt)
                        history.append(best_text)

                    rec["systems"][sysname] = {"outputs": sys_outs}

                except Exception as e:
                    # 에러를 세션/시스템 단위로 기록하고 계속 진행
                    err = f"{type(e).__name__}: {e}"
                    print(f"[ERROR] sid={sid} sys={sysname} -> {err}", file=sys.stderr)
                    traceback.print_exc()
                    rec["systems"][sysname] = {"error": err}

            # 한 세션 처리 후 즉시 쓰기 (중간에 끊겨도 파일은 채워짐)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[OK] wrote {n_written} sessions → {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(run())
