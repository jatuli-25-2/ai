# ai/src/ai_server.py
# ⛔ stateful 없음
# ⭕ Spring에서 보낸 messages 전체 기반으로 항상 동작하는 순수 생성기
import os
from typing import List, Dict
from collections import Counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BertModel

# ============================
# 환경 설정
# ============================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_FAST = "gpt-4o-mini"
MODEL_DEEP = "gpt-4o"

# 감정분석에 필요한 변수 선언
# gpu 사용 및 한국어 감정 선언
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTIONS_KO = ["분노", "혐오", "두려움", "기쁨", "중립", "슬픔", "놀람"]


# ============================
# 감정분석 모델 (KoBERT)
# ============================
class EmotionClassifier7(nn.Module):
    def __init__(self, model_path: str = "emotion_model.pt"):
        super().__init__()
        self.bert = BertModel.from_pretrained("monologg/kobert", trust_remote_code=True)
        self.fc = nn.Linear(self.bert.config.hidden_size, 7)
        state_dict = torch.load(model_path, map_location=DEVICE)
        self.load_state_dict(state_dict, strict=False)
        self.to(DEVICE)
        self.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)

    @torch.no_grad()
    def predict(self, text: str) -> Dict[str, object]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64
        ).to(DEVICE)
        outputs = self.bert(**inputs)
        cls = outputs.last_hidden_state[:, 0]
        logits = self.fc(cls)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        idx = int(probs.argmax())
        return {
            "emotion": EMOTIONS_KO[idx],   # ex) "기쁨", "슬픔" ...
            "probs": probs.tolist(),
        }


# 앱 시작 시 한 번만 로드
emotion_model = EmotionClassifier7(model_path="emotion_model.pt")


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
    emotion: str   # 마지막 사용자 발화에 대한 감정


class FinalizeRequest(BaseModel):
    mode: str
    messages: List[MessageItem]


class FinalizeResponse(BaseModel):
    finalText: str
    dominantEmotion: str   # 전체 USER 발화 중 지배적 감정
    recommend: dict            # diary: 음악 추천, book_review: 도서 추천


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
# 추천 유틸 (음악 / 도서)
# ====================================
def recommend_music(emotion: str) -> Dict[str, str]:
    """
    일기 모드: 주된 감정에 어울리는 한국 노래 추천
    """
    prompt = (
        f"'{emotion}' 감정에 어울리는 한국 노래 1곡을 추천해줘. "
        f"제목과 가수 이름을 말해주고, 왜 이 감정에 어울리는지 한두 문장 설명해줘. "
        f"유튜브 링크나 음원 링크를 하나 자연스럽게 문장 안에 포함해서 한국어 한 단락으로 작성해줘."
    )
    rec = openai_chat(MODEL_FAST, "너는 한국 음악 큐레이터야.", prompt)
    return {
        "type": "music",
        "emotion": emotion,
        "recommendation": rec,
    }


def recommend_book(emotion: str) -> Dict[str, str]:
    """
    독후감 모드: 주된 감정에 어울리는 한국 도서 추천
    """
    prompt = (
        f"'{emotion}' 감정에 어울리는 한국 도서 1권을 추천해줘. "
        f"제목, 저자, 한 줄 줄거리, 왜 이 감정에 어울리는지 설명해주고, "
        f"온라인 서점(예: 교보문고, 알라딘 등) 링크를 하나 자연스럽게 문장 안에 포함해서 "
        f"한국어 한 단락으로 작성해줘."
    )
    rec = openai_chat(MODEL_FAST, "너는 한국 도서 큐레이터야.", prompt)
    return {
        "type": "book",
        "emotion": emotion,
        "recommendation": rec,
    }


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
# 2) 다음 질문 생성 + 마지막 감정 분석
# ====================================
@app.post("/api/ai/next-question", response_model=NextQuestionResponse)
def next_question(req: NextQuestionRequest):

    # mode 검증 (diary / book_review / review)
    if req.mode not in ("diary", "book_review", "review"):
        raise HTTPException(status_code=400, detail="mode는 'diary' 또는 'book_review'/'review'만 가능합니다.")

    history = "\n".join([f"{m.role}: {m.content}" for m in req.messages])

    prompt = f"""
다음은 사용자와 AI의 대화입니다:

{history}

위 대화를 기반으로,
- 이미 했던 질문을 반복하지 말고
- 자연스럽게 이어질 다음 질문 1개만 생성하세요.
반드시 한국어 한 문장으로만 답하세요.
"""

    next_q = openai_chat(
        MODEL_FAST,
        "너는 감정 기반 한국어 인터뷰어입니다.",
        prompt
    )

    # 마지막 USER 발화에 대해 감정 분석 (추가)
    user_messages = [m.content for m in req.messages if m.role.upper() == "USER"]
    if user_messages:
        last_answer = user_messages[-1]
        emo = emotion_model.predict(last_answer)
        emotion_label = emo["emotion"]
    else:
        emotion_label = "중립"

    return NextQuestionResponse(
        nextQuestion=next_q,
        emotion=emotion_label
    )


# ====================================
# 3) 최종 글 생성 + 지배적인 감정 + 추천 (추가)
# ====================================
@app.post("/api/ai/finalize", response_model=FinalizeResponse)
def finalize(req: FinalizeRequest):

    history = "\n".join([f"{m.role}: {m.content}" for m in req.messages])

    # 모든 USER 발화에 대한 감정 분석 후, 지배적 감정 계산
    user_messages = [m.content for m in req.messages if m.role.upper() == "USER"]
    emotions: List[str] = []

    for text in user_messages:
        emo = emotion_model.predict(text)
        emotions.append(emo["emotion"])

    if emotions:
        counts = Counter(emotions)
        dominant_emotion = counts.most_common(1)[0][0]
    else:
        dominant_emotion = "중립"

    # 글 생성 프롬프트 (mode에 따라 문체 분기)
    if req.mode == "diary":
        sys_prompt = (
            "당신은 감정 기반 한국어 일기 작성 어시스턴트입니다. "
            "대화 기록과 사용자의 감정을 반영해서 따뜻하고 자연스러운 1인칭 일기를 작성하세요."
        )
    else:
        sys_prompt = (
            "당신은 감정 기반 한국어 독후감 작성 어시스턴트입니다. "
            "대화 기록과 사용자의 감정을 반영해서 서론-본론-결론 구조의 1인칭 독후감을 작성하세요."
        )

    final_text = openai_chat(MODEL_DEEP, sys_prompt, history, max_tokens=800)

    # 모드에 따라 추천 타입 분기
    if req.mode == "diary":
        rec_obj = recommend_music(dominant_emotion)
    else:
        rec_obj = recommend_book(dominant_emotion)

    return FinalizeResponse(
        finalText=final_text,
        dominantEmotion=dominant_emotion,
        recommend=rec_obj
    )
