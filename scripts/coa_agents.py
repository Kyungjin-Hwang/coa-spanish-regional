# Spanish_variable/coa_agents.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json, random, re

try:
    from .tag_lexicon import match_text  # 패키지 실행 모드
except ImportError:
    from tag_lexicon import match_text  # 직접 실행 시 fallback

# ─────────────────────────────────────────────────────────
# Planner
# ─────────────────────────────────────────────────────────
class Planner:
    def __init__(self, lexicon: Dict[str, Any], rules: Dict[str, Any]):
        self.lexicon = lexicon
        self.rules = rules

    def _pick_keywords(self, region: str, k: int = 5, register: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        사전에서 region 표지 gloss의 대표 lemma를 k개 선택.
        - register 필터가 있으면 일치 우선.
        """
        out: List[Dict[str, Any]] = []
        gmap: Dict[str, List[Dict[str, Any]]] = self.lexicon.get(region, {})
        # gloss명 알파순 → 간단/재현성
        for gloss in sorted(gmap.keys()):
            ents = gmap[gloss]
            if not ents:
                continue
            # register 우선
            ent = None
            if register:
                ent = next((e for e in ents if str(e.get("register", "")).lower() == register.lower()), None)
            ent = ent or ents[0]
            lemma = ent.get("lemma")
            if lemma:
                out.append({"gloss": gloss, "lemma": lemma, "type": ent.get("type","word"), "register": ent.get("register")})
            if len(out) >= k:
                break
        return out

    def plan(self, user_language: str, target_region: str, style: str, k_keywords: int, register: Optional[str] = None) -> Dict[str, Any]:
        r = target_region
        rule = self.rules.get(r, {})
        # prefer/ban/soft_ban은 '무가공 복사'(strip+dedup만)로 안전 전달
        def _dedup(xs):
            seen=set(); out=[]
            for s in xs or []:
                s=str(s).strip()
                if s and s not in seen:
                    seen.add(s); out.append(s)
            return out
        prefer = _dedup(rule.get("prefer", []))
        ban = _dedup(rule.get("ban", []))
        soft_ban = _dedup(rule.get("soft_ban", []))

        keywords = self._pick_keywords(r, k=k_keywords, register=register)
        return {
            "user_language": user_language,
            "target_region": r,
            "style": style,
            "register": register,
            "prefer": prefer,
            "ban": ban,
            "soft_ban": soft_ban,
            "keywords": keywords,
        }

# ─────────────────────────────────────────────────────────
# Retriever
# ─────────────────────────────────────────────────────────
class Retriever:
    def __init__(self, corpus_path: str | Path):
        self.records: List[Dict[str, Any]] = []
        p = Path(corpus_path)
        if p.exists():
            with p.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.records.append(json.loads(line))
        else:
            raise FileNotFoundError(f"corpus not found: {p}")

    def retrieve(self, policy: Dict[str, Any], k_examples: int = 3) -> List[Dict[str, Any]]:
        if k_examples <= 0:
            return []
        region = policy["target_region"]
        key_lemmas = [kw["lemma"].lower() for kw in (policy.get("keywords") or []) if kw.get("lemma")]
        prefers = set(x.lower() for x in (policy.get("prefer") or []))

        cand = []
        for rec in self.records:
            if rec.get("region") != region:
                continue
            txt = (rec.get("text") or "").lower()
            if not txt:
                continue
            hit = any(l in txt for l in key_lemmas) or any(w in txt for w in prefers)
            if hit:
                cand.append(rec)
        # fallback: region 맞는 모든 문장
        pool = cand if cand else [r for r in self.records if r.get("region")==region]
        random.shuffle(pool)
        return pool[:k_examples]

# ─────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────
class Generator:
    def __init__(self, generate_fn):
        """
        generate_fn(prompt:str, n:int)->List[str]
        """
        self.generate_fn = generate_fn

    def _build_prompt(self, policy: Dict[str, Any], examples: List[Dict[str, Any]], task_desc: str, input_text: str) -> str:
        # 간결한 지시 + 정책/예문
        lines = []
        lines.append("You are a Spanish generator that must adhere to the target regional dialect and register.")
        lines.append(f"Task: {task_desc}")
        lines.append(f"Style: {policy.get('style','neutral')}, Register: {policy.get('register') or 'default'}")
        lines.append(f"Target region: {policy['target_region']}")
        if policy.get("prefer"):
            lines.append("Prefer these regional choices (soft): " + ", ".join(policy["prefer"]))
        if policy.get("ban"):
            lines.append("Avoid (hard if auditor applies): " + ", ".join(policy["ban"]))
        if policy.get("keywords"):
            lines.append("Dialectal anchors: " + ", ".join(k['lemma'] for k in policy["keywords"]))
        if examples:
            lines.append("Few-shot examples for target region:")
            for i, ex in enumerate(examples, 1):
                lines.append(f"  ({i}) {ex['text']}")
        lines.append("Input:")
        lines.append(input_text.strip())
        lines.append("Output: Produce 1 Spanish answer in the target dialect. Keep it concise.")
        return "\n".join(lines)

    def generate(self, policy: Dict[str, Any], examples: List[Dict[str, Any]], task_desc: str, input_text: str, n: int = 6) -> List[str]:
        prompt = self._build_prompt(policy, examples, task_desc, input_text)
        outs = self.generate_fn(prompt, n=n)
        # 보정: 공백/따옴표 정리
        return [str(x).strip().strip('"').strip() for x in outs if str(x).strip()]

# ─────────────────────────────────────────────────────────
# Auditor
# ─────────────────────────────────────────────────────────
@dataclass
class Weights:
    alpha: float = 0.45   # dialect_prob
    beta:  float = 0.25   # lexicon_hit_ratio
    gamma: float = 0.20   # banned_hit_ratio
    delta: float = 0.10   # quality
    epsilon: float = 0.15 # soft_ban_ratio (감점)

class Auditor:
    def __init__(self, lexicon: Dict[str, Any], index: Dict[str, Any], rules: Dict[str, Any], weights: Dict[str, float] | None = None):
        self.lexicon = lexicon
        self.index = index
        self.rules = rules
        self.w = Weights(**(weights or {}))

        # 기본 치환 사전(룰에 replacements 없으면 사용)
        self.default_pairs = {
            "MX": {
                "ordenador": "computadora",
                "coche": "carro",
                "gafas": "lentes",
                "zumo": "jugo",
                "piscina": "alberca",
                "coger": "agarrar",
            },
            "ES": {
                "computadora": "ordenador",
                "carro": "coche",
                "lentes": "gafas",
                "jugo": "zumo",
                "alberca": "piscina",
            },
            "AR": {
                "tú": "vos",
                "carro": "auto",
                "gafas": "anteojos",
                "ordenador": "computadora",
            },
            "CL": {
                "ordenador": "computador",
                "gafas": "lentes",
            },
            "PE": {
                "ordenador": "computadora",
                "coche": "carro",
            },
            "CO": {
                "ordenador": "computador",
                "coche": "carro",
                "lentes": "gafas",  # CO는 gafas 선호
            },
        }

    def _quality_heuristic(self, text: str) -> float:
        # 매우 가벼운 휴리스틱: 길이/반복 억제
        L = len(text)
        if L > 600:
            return 0.1
        if L > 300:
            return 0.3
        # 간단 반복 토큰 페널티
        toks = re.findall(r"\w+", text.lower())
        rep = max((toks.count(t) for t in set(toks)), default=1)
        rep_pen = 0.0 if rep <= 3 else min(0.4, (rep-3)*0.05)
        return max(0.0, 0.6 - rep_pen)

    def _count_phrase_hits(self, text_l: str, items: List[str]) -> int:
        cnt = 0
        for it in items:
            s = it.lower().strip()
            if not s:
                continue
            if " " in s:  # phrase: 부분 문자열
                cnt += text_l.count(s)
            else:        # word: 경계 기반
                cnt += len(re.findall(rf"\b{s}\b", text_l))
        return cnt

    def _features(self, text: str, target_region: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        res = match_text(text, self.index)
        total = int(res["counts_by_region"].total())
        tp = int(res["counts_by_region"].get(target_region, 0))
        dialect_prob = (tp/total) if total > 0 else 0.0
        lexicon_hit_ratio = (tp/total) if total > 0 else 0.0

        # banned/soft_ban: 룰 기반 문자열 스캔
        rule = self.rules.get(target_region, {})
        text_l = " " + text.lower() + " "
        banned_hits = self._count_phrase_hits(text_l, rule.get("ban", []))
        total_tokens = max(1, len(re.findall(r"\w+", text_l)))
        banned_hit_ratio = min(1.0, banned_hits / total_tokens)

        soft_ban_hits = self._count_phrase_hits(text_l, rule.get("soft_ban", []))
        soft_ban_ratio = min(1.0, soft_ban_hits / total_tokens)

        quality = self._quality_heuristic(text)

        feats = {
            "dialect_prob": dialect_prob,
            "lexicon_hit_ratio": lexicon_hit_ratio,
            "banned_hits": int(banned_hits),
            "banned_hit_ratio": float(banned_hit_ratio),
            "soft_ban_ratio": float(soft_ban_ratio),
            "total_matches": total,
            "quality": quality,
        }
        return feats, res

    def rerank(self, candidates: List[str], target_region: str, qualities: Optional[List[float]] = None):
        scored = []
        for i, cand in enumerate(candidates):
            feats, _ = self._features(cand, target_region)
            # 외부 품질 스코어 주입(있으면 δ를 해당 값으로 대체)
            quality = qualities[i] if qualities and i < len(qualities) else feats["quality"]
            score = (self.w.alpha * feats["dialect_prob"]
                     + self.w.beta * feats["lexicon_hit_ratio"]
                     - self.w.gamma * feats["banned_hit_ratio"]
                     - self.w.epsilon * feats["soft_ban_ratio"]
                     + self.w.delta * quality)
            feats = dict(feats)
            feats["quality"] = quality
            scored.append((i, score, feats))
        scored.sort(key=lambda x: x[1], reverse=True)
        best_idx = scored[0][0] if scored else -1
        return best_idx, scored

    def hard_constrain(self, text: str, target_region: str):
        """
        금칙 발견 시 자동 치환:
          1) rules[target].replacements 우선
          2) 없으면 default_pairs[region] 사용
        """
        rule = self.rules.get(target_region, {})
        repls = list(rule.get("replacements", []))
        if not repls:
            pairs = self.default_pairs.get(target_region, {})
            repls = [{"from": k, "to": v} for k, v in pairs.items()]

        changed = False
        applied = []
        new_text = text

        for r in repls:
            src = r.get("from", "").strip()
            dst = r.get("to", "").strip()
            if not src or not dst:
                continue
            # 단어/구 별로 치환
            if " " in src:
                count = new_text.lower().count(src.lower())
                if count > 0:
                    # 대소문자 보존 없이 단순 치환
                    new_text = re.sub(re.escape(src), dst, new_text, flags=re.IGNORECASE)
                    applied.append({"from": src, "to": dst, "count": count})
                    changed = True
            else:
                # 단어 경계 유지
                pattern = re.compile(rf"\b{re.escape(src)}\b", re.IGNORECASE)
                count = len(pattern.findall(new_text))
                if count > 0:
                    new_text = pattern.sub(dst, new_text)
                    applied.append({"from": src, "to": dst, "count": count})
                    changed = True

        return new_text, changed, applied
