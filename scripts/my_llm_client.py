# -*- coding: utf-8 -*-
"""
my_llm_client.py
----------------
OpenAI API 기반 CoA Generator 호출용 모듈.
run_coa.py에서 자동으로 import되어 사용됩니다.

환경 변수:
  OPENAI_API_KEY  : 필수
  OPENAI_MODEL    : 선택 (기본값 gpt-4o-mini)
"""

from typing import List
import os
from openai import OpenAI

# ✅ 1. 클라이언트 초기화
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError(
        "❌ OPENAI_API_KEY 환경 변수가 설정되어 있지 않습니다.\n"
        "터미널에서 다음 명령으로 설정하세요:\n"
        "export OPENAI_API_KEY='sk-XXXX...'"
    )

client = OpenAI(api_key=api_key)

# ✅ 2. 기본 모델 설정
_DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ✅ 3. 함수 정의
def call_model(
    prompt: str,
    n: int = 6,
    temperature: float = 0.8,
    max_tokens: int = 256,
    model: str = _DEFAULT_MODEL,
) -> List[str]:
    """
    prompt 하나에 대해 n개의 후보를 생성하여 List[str]로 반환.
    """

    # (1) 요청 로그 (디버그용)
    print(f"[INFO] Generating {n} candidates via {model} ...")

    # (2) ChatCompletion 호출
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        print(f"[ERROR] OpenAI API 호출 실패: {e}")
        return []

    # (3) 결과 정리
    outs: List[str] = []
    for ch in getattr(resp, "choices", []):
        txt = getattr(ch.message, "content", "") or ""
        txt = txt.strip()
        if txt:
            outs.append(txt)

    print(f"[OK] {len(outs)} candidates received.")
    return outs
