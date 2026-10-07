# Spanish_variable/quality_heuristics.py
from __future__ import annotations
import re, math

def quality_score(s: str) -> float:
    if not s or not s.strip(): return 0.0
    t = s.strip()

    # 길이 페널티(토큰 4~40 권장)
    toks = re.findall(r"\w+|\S", t)
    L = len([w for w in toks if re.match(r"\w+", w)])
    len_ok = 1.0 - min(1.0, abs(L-20)/20.0)  # 20 근처가 최고

    # 반복 문자/토큰 페널티
    rep_char = 1.0 if re.search(r"(.)\1{3,}", t) else 0.0
    rep_tok  = 0.0
    words = [w.lower() for w in re.findall(r"\w+", t)]
    if words:
        most = max(words.count(w) for w in set(words))
        rep_tok = max(0.0, (most-3)/10.0)  # 3회 이상 반복 시 감점 증가

    # 구두점 밸런스(괄호/따옴표)
    def balanced(open_ch, close_ch):
        return t.count(open_ch) == t.count(close_ch)
    punct_bal = 1.0 if all([
        balanced("(",")"), balanced("[","]"), balanced("{","}"),
        t.count('"')%2==0, t.count("'")%2==0
    ]) else 0.5

    # 문장 종료부호(.?!)
    end_punct = 1.0 if re.search(r"[.?!]\s*$", t) else 0.8

    # 종합(가중 평균)
    score = (
        0.45*len_ok +
        0.20*punct_bal +
        0.15*end_punct +
        0.20*(1.0 - min(1.0, rep_char + rep_tok))
    )
    return max(0.0, min(1.0, score))
