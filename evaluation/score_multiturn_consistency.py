# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd

# --- 단순 로더(외부 의존 제거) ---
def load_lexicon(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    p = Path(path)
    if not p.exists(): return rows
    with p.open(encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                try:
                    rows.append(json.loads(ln))
                except Exception:
                    pass
    return rows

# --- 현재 스키마(지역 -> 카테고리 -> 엔트리[{lemma,forms,type}] )에서 form→regions 맵 만들기 ---
def build_form2regions_from_nested(lexicon: Dict[str, Any]) -> Dict[str, set]:
    form2regions: Dict[str, set] = {}
    for region, catdict in lexicon.items():
        if not isinstance(catdict, dict):
            continue
        for _cat, entries in catdict.items():
            if not isinstance(entries, list):
                continue
            for ent in entries:
                forms = []
                if isinstance(ent, dict):
                    # 우선 forms, 없으면 lemma
                    fs = ent.get("forms")
                    if isinstance(fs, list):
                        forms.extend([str(x) for x in fs])
                    elif isinstance(fs, str):
                        forms.append(fs)
                    lm = ent.get("lemma")
                    if isinstance(lm, str):
                        forms.append(lm)
                elif isinstance(ent, str):
                    forms.append(ent)
                # 매핑
                for w in forms:
                    w0 = str(w).strip().lower()
                    if not w0:
                        continue
                    form2regions.setdefault(w0, set()).add(region)
    return form2regions

# --- 간단한 로컬 매처: phrase 우선, 그다음 단어경계 매칭 ---
def _normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def build_match_plan(form2regions: Dict[str, set]):
    # 긴 형태(구/숙어)를 먼저 찾도록 길이 내림차순으로 정렬
    forms = sorted(form2regions.keys(), key=len, reverse=True)
    # 단어/구 구분용 간단 휴리스틱
    plan = []
    for f in forms:
        if " " in f:   # 멀티워드/숙어는 서브스트링 검색
            plan.append(("PHRASE", f))
        else:
            # 단어 경계 기반 (스페인어 알파벳, 악센트 포함 간단 처리)
            # \b 는 악센트 문자에 약해질 수 있어, 양쪽에 비문자 경계로 보강
            pattern = r"(?<!\w)" + re.escape(f) + r"(?!\w)"
            plan.append(("WORD", re.compile(pattern)))
    return plan

def match_forms(text: str, plan, form2regions: Dict[str,set]) -> List[Tuple[str,str]]:
    t = _normalize_text(text)
    hits: List[Tuple[str,str]] = []
    used_spans = []  # 겹침 필터(옵션)
    for kind, pat in plan:
        if kind == "PHRASE":
            f = pat
            idx = t.find(f)
            if idx != -1:
                # 겹침 간단 필터
                span = (idx, idx+len(f))
                if any(not (span[1]<=s or span[0]>=e) for (s,e) in used_spans):
                    continue
                used_spans.append(span)
                for rg in form2regions.get(f, set()):
                    hits.append((rg, f))
        else:  # WORD
            for m in pat.finditer(t):
                span = m.span()
                if any(not (span[1]<=s or span[0]>=e) for (s,e) in used_spans):
                    continue
                used_spans.append(span)
                f = m.group(0)
                for rg in form2regions.get(f, set()):
                    hits.append((rg, f))
    return hits

def score_session_turns(turn_texts: List[str], target_region: str,
                        plan, form2regions: Dict[str, set]) -> Dict[str, Any]:
    per_turn_shares: List[float] = []
    any_non_target = False
    all_target_only = True
    total_hits_all = 0

    for t in turn_texts:
        matches = match_forms(t, plan, form2regions)
        total = len(matches)
        total_hits_all += total
        tgt = sum(1 for (rg, _fm) in matches if rg == target_region)
        non_tgt = total - tgt
        if total > 0:
            per_turn_shares.append(tgt / total)
        if non_tgt > 0:
            any_non_target = True
            all_target_only = False

    if not per_turn_shares:
        return {
            "session_target_share_mean": 0.0,
            "session_any_nontarget": 1,
            "session_all_target_only": 0,
            "total_hits": 0,
        }
    return {
        "session_target_share_mean": float(sum(per_turn_shares)/len(per_turn_shares)),
        "session_any_nontarget": int(any_non_target),
        "session_all_target_only": int(all_target_only and (total_hits_all > 0)),
        "total_hits": int(total_hits_all),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--outputs",  required=True)
    ap.add_argument("--lexicon",  required=True)
    ap.add_argument("--out_rows", required=True)
    ap.add_argument("--out_groups", required=True)
    args = ap.parse_args()

    sess = read_jsonl(args.sessions)
    outs = read_jsonl(args.outputs)
    if not outs:
        print("[ERR] outputs is empty.")
        return

    sess_meta = {}
    for s in sess:
        sid = str(s.get("id") or s.get("session_id") or s.get("sid") or "")
        if not sid:
            continue
        sess_meta[sid] = {
            "region": s.get("region") or s.get("target_region") or s.get("region_target") or "ES",
            "turns_n": len(s.get("turns", [])),
        }

    # NEW: 현재 스키마에서 바로 form→regions 생성 + 매치 플랜 준비
    lex = load_lexicon(args.lexicon)
    form2regions = build_form2regions_from_nested(lex)
    plan = build_match_plan(form2regions)
    # print(f"[DBG] forms={len(form2regions)}")  # 필요시 확인

    rows: List[Dict[str, Any]] = []
    for o in outs:
        sid = str(o.get("id") or o.get("session_id") or o.get("sid") or "")
        sysname = str(o.get("system") or o.get("system_name") or "B1")
        turn_texts = o.get("turn_texts") or o.get("outputs") or o.get("responses") or o.get("turns") or []
        if not isinstance(turn_texts, list):
            continue
        region = o.get("region") or (sess_meta.get(sid, {}).get("region") if sid in sess_meta else None)
        if not sid or not region:
            continue

        ses = score_session_turns(turn_texts, str(region), plan, form2regions)
        rows.append({
            "id": sid, "system": sysname, "region": str(region), **ses
        })

    if not rows:
        print("[WARN] no rows built. outputs=%d" % len(outs))
        print("[HINT] keys: id/session_id/sid, system/system_name, turn_texts/outputs/turns/responses, region.")
        return

    df = pd.DataFrame(rows)
    Path(args.out_rows).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_rows, index=False, encoding="utf-8")
    print(f"[OK] row-level → {args.out_rows} ({len(df)})")

    g = df.groupby(["system","region"]).agg(
        session_target_share_mean=("session_target_share_mean","mean"),
        session_any_nontarget_rate=("session_any_nontarget","mean"),
        session_all_target_only_rate=("session_all_target_only","mean"),
        total_hits_mean=("total_hits","mean"),
        n_sessions=("id","nunique"),
    ).reset_index()
    g.to_csv(args.out_groups, index=False, encoding="utf-8")
    print(f"[OK] group-level → {args.out_groups} ({len(g)})")

if __name__ == "__main__":
    main()
