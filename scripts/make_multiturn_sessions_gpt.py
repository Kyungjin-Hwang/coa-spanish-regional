# -*- coding: utf-8 -*-
"""
make_multiturn_sessions_gpt.py
- Read a regional lexicon CSV with Spanish headers:
  España, México, Chile, Perú, Argentina, English Meaning
- Generate multi-turn dialogues per region with OpenAI (via my_llm_client).
- Keep regional lexical consistency within each session (with retries).
Writes:
  data/multiturn_sessions.jsonl
  Spanish_variable/reports/multiturn_summary.tsv
"""

import os, csv, json, re, random, argparse, pathlib
from collections import defaultdict
from typing import List, Dict

random.seed(20251014)

REGIONS = ["ES", "MX", "AR", "CL", "PE"]

# ---------- CSV 로딩 (스페인어 헤더 지원) ----------
def read_region_lexicon(csv_path: str) -> Dict[str, List[str]]:
    """
    CSV 헤더 예시:
    España,México,Chile,Perú,Argentina,English Meaning
    각 셀에 'ratero / ladrón'처럼 슬래시로 구분된 변이들이 올 수 있음.
    """
    def norm_cell(s: str) -> str:
        s = (s or "").strip()
        # 끝의 마침표 등 가벼운 문장 부호 제거
        s = re.sub(r"[\.…]+$", "", s).strip()
        # 내부 다중 공백 정리
        s = re.sub(r"\s+", " ", s)
        return s

    # 헤더명 → 지역 코드 매핑
    head2rg = {
        "españa": "ES",
        "spain": "ES",
        "méxico": "MX",
        "mexico": "MX",
        "argentina": "AR",
        "chile": "CL",
        "perú": "PE",
        "peru": "PE",
    }

    buckets = {r: [] for r in REGIONS}

    with open(csv_path, encoding="utf-8") as f:
        sample = f.read(4096); f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";"])
        except Exception:
            class _D: delimiter = ","
            dialect = _D()
        reader = csv.DictReader(f, dialect=dialect)

        # 실제 헤더명에서 지역 컬럼 식별
        region_cols = {}
        for col in (reader.fieldnames or []):
            key = (col or "").strip().lower()
            key = re.sub(r"\s+", " ", key)
            if key in head2rg:
                region_cols[head2rg[key]] = col

        for row in reader:
            for rg, col in region_cols.items():
                raw = norm_cell(row.get(col, ""))
                if not raw:
                    continue
                # 'x / y / z' 형태 분리
                parts = [norm_cell(x) for x in re.split(r"\s*/\s*|,\s*o\s*|\s* o \s*", raw) if norm_cell(x)]
                for p in parts:
                    # 너무 긴 문장은(사전이 아니라 문장일 가능성) 제외할 수도 있음
                    # 여기선 숙어도 쓰고 싶으니 6단어 이내 허용
                    if len(p.split()) > 6:
                        continue
                    if p not in buckets[rg]:
                        buckets[rg].append(p)

    # 폴백(없으면 최소 셋팅)
    defaults = {
        "ES": ["ordenador", "móvil", "mala suerte"],
        "MX": ["computadora", "celular", "estoy salado"],
        "AR": ["computadora", "celular", "mal orto"],
        "CL": ["computador", "celu", "mala cuea"],
        "PE": ["computadora", "celular", "estoy piña"],
    }
    for rg in REGIONS:
        if not buckets[rg]:
            buckets[rg] = defaults[rg]
    return buckets

# ---------- ChatGPT 호출 ----------
def call_openai_dialogue(prompt: str, n: int = 1, temperature: float = 0.4, max_tokens: int = 380) -> List[str]:
    """
    Spanish_variable/my_llm_client.call_model 사용.
    없거나 실패하면 간단 더미 대화로 폴백.
    """
    try:
        from Spanish_variable.my_llm_client import call_model as _call
        return _call(prompt, n=n, temperature=temperature, max_tokens=max_tokens)
    except Exception as e:
        return [dummy_dialogue_from_prompt(prompt)]

def dummy_dialogue_from_prompt(prompt: str) -> str:
    m = re.search(r"REGION\s*=\s*([A-Z]{2})", prompt)
    region = m.group(1) if m else "ES"
    ws = re.findall(r"- ([^\n]+)", prompt)
    # 첫 표지어만 잡아서 3턴 구성
    w = ws[0].split(",")[0].strip() if ws else ("ordenador" if region=="ES" else "computadora")
    return (
        f"Turn 1: Hola, ¿puedes ayudarme con {w}?\n"
        f"Turn 2: Claro, dime qué pasa con {w}.\n"
        f"Turn 3: Desde ayer no funciona bien.\n"
    )

# ---------- 파싱 & 일관성 검사 ----------
def parse_turns(text: str) -> List[str]:
    lines = [l.rstrip() for l in text.strip().splitlines() if l.strip()]
    turns = []
    buf = []
    for ln in lines:
        if re.match(r"(?i)turn\s*\d+\s*:", ln):
            if buf:
                turns.append(" ".join(buf).strip())
                buf = []
            ln = re.sub(r"(?i)^turn\s*\d+\s*:\s*", "", ln).strip()
            if ln:
                buf.append(ln)
        else:
            buf.append(ln)
    if buf:
        turns.append(" ".join(buf).strip())
    if len(turns) < 3:
        # 백업: 최대 5줄까지 턴으로 본다
        turns = lines[:5]
    return turns[:5]

def consistency_ok(turns: List[str], region: str, lexemes: List[str]) -> bool:
    """
    완화 버전:
    세션 일관성: 선택된 lexeme이 절반 이상의 턴에서 등장하면 일관성 유지로 간주.
    """
    if not turns:
        return False
    norm_turns = [t.lower() for t in turns]
    for lex in lexemes:
        lx = lex.lower()
        hit_count = sum(lx in t for t in norm_turns)
        if hit_count >= len(norm_turns) / 2:
            return True
    # 또는 전체 lexeme 등장합이 턴수 이상일 때도 합격
    total_hits = sum(t.count(lex.lower()) for lex in lexemes for t in norm_turns)
    return total_hits >= len(norm_turns)


# ---------- 프롬프트 ----------
def build_prompt(region: str, lexemes: List[str], turns_min: int, turns_max: int) -> str:
    return f"""You are ChatGPT 5.0 acting as a Spanish dialogue writer.

Task: Write a short multi-turn dialogue in Spanish of {turns_min} to {turns_max} turns.
Constraint: The dialogue MUST consistently reflect the regional variety of Spanish for REGION = {region}.
You MUST naturally and consistently use ONLY the following regional lexical items (avoid other regions' variants):
- {", ".join(lexemes)}

Guidelines:
- Everyday support or daily-life context; 1–2 sentences per turn; natural tone.
- Do NOT explain you are a model; do NOT discuss dialects explicitly.
- Keep the chosen lexemes consistent across turns (no mixing with other regions).
- Output strictly in this format:

Turn 1: ...
Turn 2: ...
Turn 3: ...
(Turn 4/5 if needed)
"""

# ---------- 메인 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="e.g., /Users/kyungjin/PycharmProjects/LargeLanguageModel/Spanish_variable/Regional_Variant_Dictionary_370rows.csv")
    ap.add_argument("--sessions_per_region", type=int, default=10)
    ap.add_argument("--lexemes_per_session", type=int, default=2)
    ap.add_argument("--turns_min", type=int, default=3)
    ap.add_argument("--turns_max", type=int, default=5)
    ap.add_argument("--retries", type=int, default=2, help="consistency 실패 시 재시도")
    ap.add_argument("--out_jsonl", default="data/multiturn_sessions.jsonl")
    ap.add_argument("--out_report", default="Spanish_variable/reports/multiturn_summary.tsv")
    args = ap.parse_args()

    pathlib.Path("data").mkdir(exist_ok=True, parents=True)
    pathlib.Path("Spanish_variable/reports").mkdir(exist_ok=True, parents=True)

    lex = read_region_lexicon(args.csv)

    rows = []
    by_region = defaultdict(int)
    consistent_cnt = defaultdict(int)

    sid = 0
    for rg in REGIONS:
        pool = lex.get(rg, [])
        if not pool:
            continue
        for _ in range(args.sessions_per_region):
            sid += 1
            # 세션당 표지어 샘플링(중복 방지)
            k = min(args.lexemes_per_session, len(pool))
            sample = random.sample(pool, k=k)

            prompt = build_prompt(rg, sample, args.turns_min, args.turns_max)

            final_turns = []
            tried = 0
            while tried <= args.retries:
                tried += 1
                outs = call_openai_dialogue(prompt, n=1, temperature=0.5, max_tokens=380)
                text = outs[0] if outs else ""
                turns = [t for t in parse_turns(text) if t]
                if len(turns) < args.turns_min:
                    continue
                if consistency_ok(turns, rg, sample):
                    final_turns = turns
                    break

            if not final_turns:
                final_turns = turns if turns else ["Hola.", "¿Qué tal?", "Adiós."]

            rows.append({
                "id": f"mt_{sid:04d}",
                "region": rg,
                "seed_lexemes": sample,
                "turns": final_turns
            })
            by_region[rg] += 1
            if consistency_ok(final_turns, rg, sample):
                consistent_cnt[rg] += 1

    # 저장
    outp = pathlib.Path(args.out_jsonl)
    outp.parent.mkdir(exist_ok=True, parents=True)
    with outp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 요약
    with open(args.out_report, "w", encoding="utf-8") as rf:
        rf.write("region\tsessions\tconsistent\tratio\n")
        for rg in REGIONS:
            tot = by_region[rg]
            con = consistent_cnt[rg]
            ratio = (con / tot) if tot else 0.0
            rf.write(f"{rg}\t{tot}\t{con}\t{ratio:.3f}\n")

    print(f"[OK] wrote {len(rows)} sessions → {args.out_jsonl}")
    print(f"[OK] summary → {args.out_report}")

if __name__ == "__main__":
    main()
