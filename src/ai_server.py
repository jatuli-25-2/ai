# ai/src/ai_server.py
# ⛔ stateful 없음
# ⭕ Spring에서 보낸 messages 전체 기반으로 항상 동작하는 순수 생성기
import os

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_FAST = "gpt-4o-mini"
MODEL_DEEP = "gpt-4o"

# ====================================
#  Pydantic Models
# ====================================
class MessageItem(BaseModel):
    role: str   # "AI" or "USER"
    content: str

class NextQuestionRequest(BaseModel):
    mode: str
    messages: List[MessageItem]

class NextQuestionResponse(BaseModel):
    nextQuestion: str
    emotion: str

class FinalizeRequest(BaseModel):
    mode: str
    messages: List[MessageItem]

class FinalizeResponse(BaseModel):
    finalText: str
    dominantEmotion: str
    music: dict


app = FastAPI()


# ====================================
# OpenAI 공통 함수
# ====================================
def openai_chat(model, system, user, max_tokens=400):
    res = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return res.choices[0].message.content.strip()


# ====================================
# 1) 첫 질문
# ====================================
@app.get("/api/ai/start")
def get_first_question(mode: str):
    if mode == "diary":
        question = "오늘 하루 중 가장 기억에 남는 순간은 무엇이었나요?"
    else:
        question = "최근 읽은 책은 무엇이며, 선택한 이유는 무엇인가요?"
    return {"question": question}


# ====================================
# 2) 다음 질문 생성
# ====================================
@app.post("/api/ai/next-question", response_model=NextQuestionResponse)
def next_question(req: NextQuestionRequest):

    history = "\n".join([f"{m.role}: {m.content}" for m in req.messages])

    prompt = f"""
다음은 사용자와 AI의 대화입니다:

{history}

위 대화를 기반으로,
- 이미 했던 질문을 반복하지 말고
- 자연스럽게 이어질 다음 질문 1개만 생성하세요.
"""

    next_q = openai_chat(
        MODEL_FAST,
        "너는 감정 기반 한국어 인터뷰어입니다.",
        prompt
    )

    # 감정은 여기서는 neutral로 고정
    return NextQuestionResponse(
        nextQuestion=next_q,
        emotion="neutral"
    )


# ====================================
# 3) 최종 글 생성
# ====================================
@app.post("/api/ai/finalize", response_model=FinalizeResponse)
def finalize(req: FinalizeRequest):

    history = "\n".join([f"{m.role}: {m.content}" for m in req.messages])

    sys_prompt = (
        "당신은 감정 기반 글쓰기 어시스턴트입니다. "
        "대화 기록을 기반으로 사용자의 하루 또는 독후감을 정리하여 "
        "자연스러운 글을 완성하세요."
    )

    final_text = openai_chat(MODEL_DEEP, sys_prompt, history, max_tokens=800)

    # 음악 추천
    music_prompt = "기분을 차분하게 하는 한국 노래 한 곡과 링크를 추천해줘."
    music = openai_chat(MODEL_FAST, "너는 음악 큐레이터", music_prompt)

    return FinalizeResponse(
        finalText=final_text,
        dominantEmotion="neutral",
        music={"recommendation": music}
    )