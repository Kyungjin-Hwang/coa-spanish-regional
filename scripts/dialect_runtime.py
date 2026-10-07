# Spanish_variable/dialect_runtime.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
import re

try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None

# tag_lexicon 유틸
# --- at the top of runs/dialect_runtime.py ---


# 기존: from tag_lexicon import match_text, _norm  # ← 이게 문제
# 교체:
try:
    # 패키지 모드에서
    from .tag_lexicon import match_text, _norm
except Exception:
    try:
        # 혹시 다른 위치에서 패키지로 불릴 때
        from runs.tag_lexicon import match_text, _norm
    except Exception:
        # 스크립트 단독 실행일 때
        from tag_lexicon import match_text, _norm



# ─────────────────────────────────────────────
# 규칙 로딩
# ─────────────────────────────────────────────
def load_rules(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"rules file not found: {p}")
    if p.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError("pyyaml not installed; cannot read YAML")
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    # json (fallback)
    return json.loads(p.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────
# 정규화 유틸: 주석 제거 + 중복 제거
# ─────────────────────────────────────────────
_COMMENT_BRACKET_RE = re.compile(r"\s*\[[^\]]*\]\s*$")  # 끝의 [ ... ] 주석
_COMMENT_PAREN_RE   = re.compile(r"\s*\([^)]*\)\s*$")   # 끝의 ( ... ) 주석

def _clean_token(s: str) -> str:
    """문자열 끝의 괄호/대괄호 주석을 제거하고 트림."""
    s = str(s or "").strip()
    # 예: "coger (부정적 의미)" → "coger"
    s = _COMMENT_BRACKET_RE.sub("", s)
    s = _COMMENT_PAREN_RE.sub("", s)
    return s.strip()

def _dedup_str_list(xs: List[str]) -> List[str]:
    """
    - _clean_token을 적용
    - casefold 기반으로 중복 제거 (악센트/대소문자 차이에 덜 민감)
    - 빈 문자열 제거
    """
    out: List[str] = []
    seen: set[str] = set()
    for x in xs or []:
        s = _clean_token(x)
        if not s:
            continue
        key = s.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


# ─────────────────────────────────────────────
# 규칙 정규화
# ─────────────────────────────────────────────
def normalize_rules(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    표준화:
      - prefer/ban/soft_ban: 문자열 리스트 보장 + 주석 제거 + 중복 제거
      - ban 우선 규칙: ban에 들어간 토큰은 prefer에서 제거
      - replacements: [{from,to}] 클린업(주석 제거)
      - notes: 임의 문자열(없으면 None)
    """
    norm: Dict[str, Any] = {}
    for region, spec in (raw or {}).items():
        obj = dict(spec or {})
        prefer   = _dedup_str_list(obj.get("prefer", []))
        ban      = _dedup_str_list(obj.get("ban", []))
        soft_ban = _dedup_str_list(obj.get("soft_ban", []))

        # ban 우선: prefer에서 ban과 충돌하는 항목 제거
        ban_keys = {_norm(b) for b in ban}
        prefer = [s for s in prefer if _norm(s) not in ban_keys]

        # replacements 정리(주석 제거)
        repl_raw = obj.get("replacements", []) or []
        replacements: List[Dict[str, str]] = []
        for r in repl_raw:
            try:
                f = _clean_token(r.get("from", ""))
                t = _clean_token(r.get("to", ""))
            except Exception:
                f = t = ""
            if f and t:
                replacements.append({"from": f, "to": t})

        norm[region] = {
            "prefer": prefer,
            "ban": ban,
            "soft_ban": soft_ban,
            "replacements": replacements,
            "notes": obj.get("notes"),
        }
    return norm


# ─────────────────────────────────────────────
# 간단 룰 분류기 (사전 hit 기반)
# ─────────────────────────────────────────────
def classify_region_rule(text: str, index: dict) -> Tuple[Optional[str], dict]:
    """
    match_text 결과의 counts_by_region에서 최다 매치 지역을 예측으로 반환.
    동률이면 None.
    """
    res = match_text(text, index)
    counts = res.get("counts_by_region", {})
    if not counts:
        return None, res
    max_c = max(counts.values())
    winners = [r for r, c in counts.items() if c == max_c]
    pred = winners[0] if len(winners) == 1 else None
    return pred, res


# ─────────────────────────────────────────────
# Lexicon/Auditor 피처
# ─────────────────────────────────────────────
def compute_lexicon_features(
    text: str,
    target_region: str,
    index: dict,
    rules: Dict[str, Any],
    match_result: Optional[dict] = None,
) -> dict:
    res = match_result or match_text(text, index)
    matches = res.get("matches", [])
    total = len(matches)

    # 목표지역 히트
    target_hits = sum(1 for m in matches if m.get("region") == target_region)
    lexicon_hit_ratio = (target_hits / total) if total > 0 else 0.0

    # 금칙/소프트금칙
    r = rules.get(target_region, {})
    banned_set = {_norm(x) for x in r.get("ban", [])}
    soft_set   = {_norm(x) for x in r.get("soft_ban", [])}

    banned_hits = 0
    soft_hits = 0
    for m in matches:
        lem = _norm(m.get("lemma", ""))
        frm = _norm(m.get("form", ""))
        if lem in banned_set or frm in banned_set:
            banned_hits += 1
        elif lem in soft_set or frm in soft_set:
            soft_hits += 1

    banned_hit_ratio = (banned_hits / total) if total > 0 else 0.0
    soft_ban_ratio   = (soft_hits / total) if total > 0 else 0.0

    # 지역 확률(정규화 수치)
    counts = res.get("counts_by_region", {})
    sum_all = sum(counts.values()) or 1
    dialect_prob = counts.get(target_region, 0) / sum_all

    return {
        "dialect_prob": dialect_prob,
        "lexicon_hit_ratio": lexicon_hit_ratio,
        "banned_hits": banned_hits,
        "banned_hit_ratio": banned_hit_ratio,
        "soft_ban_ratio": soft_ban_ratio,
        "total_matches": total,
    }


# ─────────────────────────────────────────────
# Auditor 점수
# Score = α·DialectProb + β·LexiconHit − γ·BannedHit − ε·SoftBan + δ·Quality
# ─────────────────────────────────────────────
def auditor_score(
    text: str,
    target_region: str,
    index: dict,
    rules: dict,
    quality_score: float = 0.0,
    weights: dict | None = None,
) -> Tuple[float, dict, dict]:
    if weights is None:
        # 기본값(실험용)
        weights = {"alpha": 0.45, "beta": 0.25, "gamma": 0.20, "delta": 0.10, "epsilon": 0.15}
    res = match_text(text, index)
    feats = compute_lexicon_features(text, target_region, index, rules, res)
    # 외부 품질 점수 포함
    feats["quality"] = float(quality_score or 0.0)
    score = (
        weights["alpha"] * feats["dialect_prob"]
        + weights["beta"] * feats["lexicon_hit_ratio"]
        - weights["gamma"] * feats["banned_hit_ratio"]
        - weights["epsilon"] * feats["soft_ban_ratio"]
        + weights["delta"] * feats["quality"]
    )
    return float(score), feats, res


# ─────────────────────────────────────────────
# n-best 재랭크
# ─────────────────────────────────────────────
def rerank_candidates(
    candidates: List[str],
    target_region: str,
    index: dict,
    rules: dict,
    qualities: Optional[List[float]] = None,
    weights: dict | None = None,
) -> Tuple[int, List[tuple]]:
    if qualities is None:
        qualities = [0.0] * len(candidates)
    scored: List[tuple] = []
    for i, (cand, q) in enumerate(zip(candidates, qualities)):
        s, feats, _ = auditor_score(cand, target_region, index, rules, q, weights)
        scored.append((i, float(s), feats))
    scored.sort(key=lambda x: x[1], reverse=True)
    best_idx = scored[0][0] if scored else -1
    return best_idx, scored


# ─────────────────────────────────────────────
# CoA-H: 금칙 위반 자동 치환 (케이스 보존 간단판)
# ─────────────────────────────────────────────
def _preserve_case(src: str, repl: str) -> str:
    # 단어 전체 대문자
    if src.isupper():
        return repl.upper()
    # 단어 전체 소문자
    if src.islower():
        return repl.lower()
    # 첫 글자만 대문자(Title-ish)
    if len(src) > 1 and src[0].isupper() and src[1:].islower():
        return repl[:1].upper() + repl[1:].lower()
    # 토큰 단위로 최대한 매칭
    s_tok = re.split(r"(\W+)", src)
    r_tok = re.split(r"(\W+)", repl)
    out = []
    j = 0
    for t in s_tok:
        if re.match(r"\W+$", t):
            out.append(t)
        else:
            if j < len(r_tok):
                cand = r_tok[j]
                # 단어 토큰만 케이스 적용
                if re.match(r"\w+", cand):
                    out.append(_preserve_case(t.lower(), cand))
                else:
                    out.append(cand)
                j += 1
            else:
                out.append(t)
    # 남은 토큰 붙이기
    if j < len(r_tok):
        out.extend(r_tok[j:])
    return "".join(out)


def hard_constrain_replace(
    text: str,
    target_region: str,
    index: dict,
    rules: dict,
    lexicon: dict,
) -> Tuple[str, bool, list]:
    """
    - rules[target].replacements 우선 적용: span을 직접 찾기 어렵기 때문에 단순 문자열치환이 아니라
      match_text로 매칭된 span들 중 금칙/치환 대상만 바꾼다.
    - 그 외에는 매치 메타(gloss)를 이용해 타깃 지역의 대안 lemma로 치환(가능한 경우).
    """
    res = match_text(text, index)
    banned = {_norm(x) for x in rules.get(target_region, {}).get("ban", [])}

    # from->to 매핑(지역 규칙 기반)
    repl_rules = rules.get(target_region, {}).get("replacements", []) or []
    rules_map = { _norm(r["from"]): r["to"] for r in repl_rules if r.get("from") and r.get("to") }

    replacements = []
    spans = []
    for m in res.get("matches", []):
        lem = _norm(m.get("lemma", ""))
        frm = _norm(m.get("form", ""))
        gloss = m.get("gloss")
        s, e = m.get("span", [None, None])

        # 1) 명시적 replacements 적용
        key = lem or frm
        if key in rules_map and s is not None:
            repl = rules_map[key]
            spans.append((s, e, repl, m))
            continue

        # 2) 금칙이면 gloss로 대안 찾기
        if (lem in banned or frm in banned) and s is not None:
            tgt_entries = (lexicon.get(target_region, {}).get(gloss) or [])
            if tgt_entries:
                repl = tgt_entries[0].get("lemma") or tgt_entries[0].get("form") or ""
                if repl:
                    spans.append((s, e, repl, m))

    if not spans:
        return text, False, replacements

    # 역순으로 치환(인덱스 안정)
    spans.sort(key=lambda x: x[0], reverse=True)
    out = text
    for s, e, repl, meta in spans:
        src_seg = out[s:e]
        out = out[:s] + _preserve_case(src_seg, repl) + out[e:]
        replacements.append({"from": src_seg, "to": repl, "meta": meta})
    return out, True, replacements


# ─────────────────────────────────────────────
# 편의 로더 (단독 실행용)
# ─────────────────────────────────────────────
def load_runtime(
    lexicon_path: str | Path = "lexicons_out/lexicon_combined.json",
    rules_path: str | Path = "lexicons_out/rules.yml",
    try_add_rules_to_index: bool = True,
):
    # lexicon/index는 tag_lexicon 포맷을 따름
    from tag_lexicon import load_lexicon, build_index
    lexicon = load_lexicon(lexicon_path)
    index = build_index(lexicon)
    rules = normalize_rules(load_rules(rules_path))

    # 선택: rules의 prefer를 인덱스에 보강(사전에 없어도 매칭되게)
    if try_add_rules_to_index:
        try:
            from tag_lexicon import add_rules_to_index  # 선택 기능
            add_rules_to_index(index, rules)
        except Exception:
            pass
    return lexicon, index, rules
