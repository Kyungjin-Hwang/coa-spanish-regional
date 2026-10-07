# tag_lexicon.py
from __future__ import annotations
import json, re, argparse
from pathlib import Path
from collections import defaultdict, Counter

# ─────────────────────────────────────────────────────────
# 선택: spaCy 레mmatizer 사용 (없으면 자동 폴백)
# ─────────────────────────────────────────────────────────
try:
    import spacy
    try:
        nlp = spacy.load("es_core_news_sm")
    except OSError:
        # 모델이 없으면 pipeline만 빈껍데기 + 간단 tokenizer
        nlp = spacy.blank("es")
except Exception:
    nlp = None

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def load_lexicon(path_json: str) -> dict:
    data = json.load(open(path_json, encoding="utf-8"))
    # data: { "ES": {gloss: [ {lemma, forms, type, register?}, ...], ...}, "MX": ...}
    return data

def build_index(lexicon: dict) -> dict:
    """
    region별로 word/phrase 용 인덱스 구성
    - word_forms[region]: {form -> [(gloss, lemma, register)]}
    - lemma_index[region]: {lemma -> [(gloss, register)]}
    - phrase_forms[region]: [(form, gloss, lemma, register)]  # form이 공백/기호 포함
    """
    word_forms = defaultdict(lambda: defaultdict(list))
    lemma_index = defaultdict(lambda: defaultdict(list))
    phrase_forms = defaultdict(list)

    for region, gmap in lexicon.items():
        for gloss, entries in gmap.items():
            for e in entries:
                lemma = _norm(e.get("lemma",""))
                reg = e.get("register")
                etype = e.get("type", "word")
                forms = e.get("forms", []) or ([lemma] if lemma else [])
                if etype == "phrase" or any(" " in f for f in forms) or any(re.search(r"[^\wáéíóúñü]", f) for f in forms):
                    # 구/문장 또는 비단어문자 포함 → phrase로 처리(그대로 부분 문자열 검색)
                    for f in forms:
                        f_ = _norm(f)
                        if f_:
                            phrase_forms[region].append((f_, gloss, lemma, reg))
                else:
                    # 단어: 정확 토큰/lemma 매칭
                    if lemma:
                        lemma_index[region][lemma].append((gloss, reg))
                    for f in forms:
                        f_ = _norm(f)
                        if f_:
                            word_forms[region][f_].append((gloss, lemma, reg))

    return {
        "word_forms": word_forms,
        "lemma_index": lemma_index,
        "phrase_forms": phrase_forms,
        "regions": list(lexicon.keys())
    }

def tokenize_lemmas(text: str):
    text = text.strip()
    if not text:
        return []
    if nlp is None:
        # 폴백: 공백 분리 + 소문자, lemma = token
        toks = _norm(text).split()
        return [(t, t) for t in toks]
    doc = nlp(text)
    out = []
    for t in doc:
        tok = t.text.lower()
        # lemma가 없으면 소문자 토큰을 lemma로 사용
        lem = (t.lemma_ or tok).lower()
        out.append((tok, lem))
    return out

def match_text(text: str, index: dict):
    """
    텍스트에서 지역별 매칭 수행
    - phrase 먼저(부분문자열), 그 다음 word(토큰/lemma)
    - 겹침 방지: phrase로 잡힌 구간은 word 매칭에서 제외
    반환:
      matches: [ {region, gloss, form, lemma, register, span=(start,end), type} ... ]
      counts_by_region: Counter
      region_hit: {region: 0/1}
    """
    regions = index["regions"]
    word_forms = index["word_forms"]
    lemma_index = index["lemma_index"]
    phrase_forms = index["phrase_forms"]

    text_l = " " + _norm(text) + " "
    taken = [False] * len(text_l)  # 겹침 방지 마스크
    matches = []

    def mark_span(s,e):
        for i in range(max(0,s), min(len(text_l), e)):
            taken[i] = True

    # 1) PHRASE 매칭 (부분 문자열)
    for region in regions:
        for form, gloss, lemma, reg in phrase_forms[region]:
            # form이 그대로 들어있는지 검사 (소문자 기준)
            start = 0
            while True:
                pos = text_l.find(form, start)
                if pos == -1: break
                end = pos + len(form)
                # 이미 커버된 영역이면 skip
                if any(taken[pos:end]):
                    start = pos + 1
                    continue
                matches.append({
                    "region": region, "gloss": gloss, "form": form, "lemma": lemma,
                    "register": reg, "span": (pos, end), "type": "phrase"
                })
                mark_span(pos, end)
                start = end

    # 2) WORD 매칭 (토큰/lemma)
    toks = tokenize_lemmas(text)
    # text_l 기준으로 토큰의 start/end 잡기(간단 추정)
    # 안전하게 하려면 spacy 토큰 span 사용; 여기선 간단히 find로 진행
    cursor = 0
    for tok, lem in toks:
        tok_ = " " + tok + " "
        pos = text_l.find(tok_, cursor)
        if pos == -1:
            # fallback: 아무데나 검색
            pos = text_l.find(tok_)
        if pos == -1:
            # 못 찾으면 스킵
            continue
        start = pos + 1
        end = start + len(tok)
        cursor = end

        # phrase로 이미 커버된 영역이면 skip
        if any(taken[start:end]):
            continue

        tok_norm = _norm(tok)
        lem_norm = _norm(lem)

        # form → region hit
        for region in regions:
            hit = False
            if tok_norm in word_forms[region]:
                for gloss, lemma_saved, reg in word_forms[region][tok_norm]:
                    matches.append({
                        "region": region, "gloss": gloss, "form": tok_norm, "lemma": lemma_saved,
                        "register": reg, "span": (start, end), "type": "word"
                    })
                    hit = True
            # lemma로도 매칭(동일 단어군)
            if lem_norm in lemma_index[region]:
                for gloss, reg in lemma_index[region][lem_norm]:
                    matches.append({
                        "region": region, "gloss": gloss, "form": tok_norm, "lemma": lem_norm,
                        "register": reg, "span": (start, end), "type": "word(lemma)"
                    })
                    hit = True
            if hit:
                # 중복 추가로 여러 지역이 같은 토큰을 잡을 수 있음 (예: 중립어휘)
                pass

    # 집계
    counts_by_region = Counter(m["region"] for m in matches)
    region_hit = {r: int(counts_by_region.get(r,0) > 0) for r in regions}

    return {"matches": matches, "counts_by_region": counts_by_region, "region_hit": region_hit}

# ─────────────────────────────────────────────────────────
# Precision / Recall 계산
#  - Precision = 목표지역 단어 / 전체 지역 단어
#  - Recall = 목표지역 단어 / "해당 문장에서 목표지역으로 쓸 기회(opportunities)"
#    → 기회 정의: 문장 내에서 발견된 각 gloss 발생 중, 그 gloss에 목표지역 변이가 존재하는 발생 수
# ─────────────────────────────────────────────────────────
def compute_pr_re(result: dict, lexicon: dict, target_region: str):
    matches = result["matches"]
    total = len(matches)
    tp = sum(1 for m in matches if m["region"] == target_region)

    precision = (tp / total) if total > 0 else None

    # 기회(opportunities): 해당 문장에서 발견된 모든 match 발생 중,
    # 그 발생의 gloss가 target_region에도 등록되어 있으면 1 기회로 카운트
    opportunities = 0
    for m in matches:
        gloss = m["gloss"]
        if target_region in lexicon and gloss in lexicon[target_region]:
            opportunities += 1

    recall = (tp / opportunities) if opportunities > 0 else None

    # (옵션) unique-gloss 기준도 함께 계산
    glosses_in_text = [m["gloss"] for m in matches]
    unique_glosses = set(glosses_in_text)
    unique_tp_gloss = set(m["gloss"] for m in matches if m["region"] == target_region)
    unique_opportunities = set(g for g in unique_glosses
                               if target_region in lexicon and g in lexicon[target_region])
    recall_unique = (len(unique_tp_gloss) / len(unique_opportunities)) if len(unique_opportunities) > 0 else None

    return {
        "precision": precision,
        "recall": recall,
        "recall_unique": recall_unique,
        "tp": tp,
        "total_matches": total,
        "opportunities": opportunities,
        "unique_tp_gloss": len(unique_tp_gloss),
        "unique_opportunities": len(unique_opportunities)
    }

# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", default="lexicons_out/lexicon_combined.json")
    ap.add_argument("--text", type=str, required=True, help="평가할 문장/텍스트")
    ap.add_argument("--target_region", type=str, default="MX", help="정밀도/재현율 계산 대상 지역 코드")
    args = ap.parse_args()

    lexicon = load_lexicon(args.lexicon)
    index = build_index(lexicon)
    res = match_text(args.text, index)

    print("== region_hit ==", res["region_hit"])
    print("== counts_by_region ==", dict(res["counts_by_region"]))
    pr = compute_pr_re(res, lexicon, args.target_region.upper())
    print("== PR ==", pr)

    # 매치 샘플 10개만 보기
    for m in res["matches"][:10]:
        print(m)

# --- add: rules boost into index ---------------------------------
# Spanish_variable/tag_lexicon.py (일부만)
def add_rules_to_index(index: dict, rules: dict, include_softban: bool = True) -> None:
    """
    rules의 prefer(+선택적으로 soft_ban)를 인덱스에 보강.
    - 중복 방지: 동일 (token, region, lemma) 조합은 1회만
    - gloss에 출처 표시: 'rule_prefer' / 'rule_soft_ban'
    """
    def _put(token: str, region: str, gloss_tag: str):
        key = _norm(token)  # 기존 match_text에서 쓰는 정규화와 동일해야 함
        bucket = index.setdefault(key, [])
        # 중복 방지
        for e in bucket:
            if e.get("region") == region and _norm(e.get("lemma","")) == _norm(token):
                return
        bucket.append({
            "region": region,
            "lemma": token,
            "form": token,
            "gloss": gloss_tag,
        })

    for region, spec in (rules or {}).items():
        for w in spec.get("prefer", []) or []:
            _put(w, region, "rule_prefer")
        if include_softban:
            for w in spec.get("soft_ban", []) or []:
                _put(w, region, "rule_soft_ban")




if __name__ == "__main__":
    main()
