import os
from uuid import uuid4
from collections import Counter
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, AutoTokenizer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel
from openai import OpenAI
import pandas as pd

# ============================
# 환경 설정
# ============================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_FAST = "gpt-4o-mini"
MODEL_DEEP = "gpt-4o"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTION_LABELS = ["happiness", "sadness", "angry", "disgust", "fear", "surprise", "neutral"]

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
GOOD_Q_PATH = os.path.join(LOG_DIR, "good_questions.csv")

TARGET_INITIAL_QA = 5   # 기본 질의응답 개수

# ============================
# KoBERT 감정분석 모델
# ============================

class EmotionClassifier7(nn.Module):
    def __init__(self, model_path: str = "emotion_model.pt"):
        super().__init__()
        self.bert = BertModel.from_pretrained("monologg/kobert", trust_remote_code=True)
        self.fc = nn.Linear(self.bert.config.hidden_size, len(EMOTION_LABELS))
        state_dict = torch.load(model_path, map_location=DEVICE)
        self.load_state_dict(state_dict, strict=False)
        self.to(DEVICE)
        self.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)

    @torch.no_grad()
    def predict(self, text: str) -> Dict:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        ).to(DEVICE)

        outputs = self.bert(**encoded)
        cls = outputs.last_hidden_state[:, 0]
        logits = self.fc(cls)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        idx = int(probs.argmax())
        return {
            "emotion": EMOTION_LABELS[idx],
            "probs": probs.tolist(),
        }


emotion_model = EmotionClassifier7(model_path="emotion_model.pt")

# ============================
# 세션 관리 구조
# ============================

class QAPair(BaseModel):
    question: str
    answer: str
    emotion: str
    probs: List[float]
    is_extra: bool = False


class WritingSession:
    def __init__(self, mode: str):
        self.mode = mode               # "diary" or "review"
        self.qa: List[QAPair] = []
        self.current_question: Optional[str] = None
        self.finished_initial: bool = False
        self.last_text: Optional[str] = None
        self.satisfied: Optional[bool] = None

        self.extra_target: int = 0     # 추가 질문 목표 개수 (addN)
        self.extra_done: int = 0


sessions: Dict[str, WritingSession] = {}

# ============================
# Pydantic 요청/응답 모델
# ============================

class StartRequest(BaseModel):
    mode: str  # "diary" / "review"


class StartResponse(BaseModel):
    sessionId: str
    question: str
    step: int
    targetInitial: int


class AnswerRequest(BaseModel):
    sessionId: str
    answer: str
    likeQuestion: Optional[bool] = None   # 사용자가 "좋은 질문"이라고 표시할지 여부 (선택)


class AnswerResponse(BaseModel):
    sessionId: str
    emotion: str
    probs: List[float]
    totalQA: int
    needMoreQA: bool
    canGenerateResult: bool
    nextQuestion: Optional[str] = None


class ResultRequest(BaseModel):
    sessionId: str


class ResultResponse(BaseModel):
    sessionId: str
    finalText: str
    dominantEmotion: str
    music: Dict[str, str]


class FeedbackRequest(BaseModel):
    sessionId: str
    satisfied: bool
    addN: Optional[int] = 0


class FeedbackResponse(BaseModel):
    sessionId: str
    status: str         # "done" or "need_more_qa"
    question: Optional[str] = None
    totalQA: Optional[int] = None


# ============================
# 공통 유틸
# ============================

def openai_chat(model_name: str, system: str, user: str, max_tokens: int = 400) -> str:
    try:
        res = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI API Error:", e)
        return "죄송합니다. 잠시 후 다시 시도해주세요."


def get_good_questions(mode: str, top_n: int = 5) -> List[str]:
    if not os.path.exists(GOOD_Q_PATH):
        return []
    df = pd.read_csv(GOOD_Q_PATH)
    df_mode = df[df["모드"] == mode]
    return df_mode.tail(top_n)["질문"].tolist()


def save_good_question(question: str, mode: str):
    row = pd.DataFrame(
        [{"질문": question, "모드": mode, "등록일": pd.Timestamp.now().strftime("%Y-%m-%d")}]
    )
    if os.path.exists(GOOD_Q_PATH):
        row.to_csv(GOOD_Q_PATH, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        row.to_csv(GOOD_Q_PATH, index=False, encoding="utf-8-sig")


def build_qa_history(session: WritingSession) -> str:
    parts = []
    for qa in session.qa:
        parts.append(f"Q: {qa.question}\nA: {qa.answer} (감정: {qa.emotion})")
    return "\n\n".join(parts)


def get_dominant_emotion(session: WritingSession) -> str:
    emotions = [qa.emotion for qa in session.qa if qa.emotion]
    if not emotions:
        return "neutral"
    counter = Counter(emotions)
    return counter.most_common(1)[0][0]


def recommend_music(emotion: str) -> Dict[str, str]:
    prompt = (
        f"비교적 유명한 한국 노래 중 '{emotion}' 감정에 어울리는 노래 1곡과 "
        f"유튜브 링크를 자연스럽게 포함한 한 문단으로 추천해줘."
    )
    res = openai_chat(MODEL_FAST, "너는 한국 음악 큐레이터야.", prompt)
    return {"emotion": emotion, "recommendation": res}


def generate_initial_question(mode: str) -> str:
    if mode == "diary":
        return "오늘 하루 어땠나요?"
    else:
        return "이 책을 읽게 된 이유는 무엇인가요?"


def generate_next_question(session: WritingSession, is_extra: bool = False) -> str:
    qa_history = build_qa_history(session)
    emotion_trend = [qa.emotion for qa in session.qa]
    good_examples = get_good_questions(session.mode)
    good_examples_text = "\n".join(f"- {q}" for q in good_examples) if good_examples else "없음"

    previous_questions = [qa.question for qa in session.qa]

    if session.mode == "diary":
        sys_prompt = "너는 감정을 읽으며 대화를 이끄는 따뜻한 일기 코치입니다."
        goal_text = (
            "감정의 배경이나 이유를 탐색하거나, 하루의 긍정적인 측면과 배운 점을 이끌어내는 "
            "질문을 한 개 만들어주세요."
        )
    else:
        sys_prompt = "너는 감정 중심의 독후감 인터뷰어이며, 감정과 통찰을 연결하는 질문을 잘합니다."
        goal_text = (
            "작품의 주제나 교훈, 인물의 감정 변화를 더 깊게 탐구하는 질문을 한 개 만들어주세요."
        )

    extra_text = " (현재는 추가 질문 단계입니다.)" if is_extra else ""

    user_prompt = f"""
지금까지의 대화:
{qa_history}

사용자의 감정 변화 흐름: {emotion_trend}

지금까지 했던 질문 목록:
{previous_questions}

사람들이 좋다고 평가한 질문 예시:
{good_examples_text}

위 내용을 참고해서{extra_text}
- 예시 질문이나 이전에 했던 질문을 그대로 반복하지 말고,
- 너무 추상적인 질문은 피하고, 구체적인 상황을 떠올리게 하는 질문으로,
{goal_text}

반드시 한국어 한 문장으로만 대답하세요.
"""

    return openai_chat(MODEL_FAST, sys_prompt, user_prompt)


def generate_result_text(session: WritingSession) -> str:
    qa_block = build_qa_history(session)

    if session.mode == "diary":
        sys_prompt = (
            "당신은 감정에 공감하는 한국어 일기 작성 도우미입니다. "
            "사용자의 답변과 감정을 바탕으로 따뜻하고 진솔한 1인칭 일기를 작성하세요. "
            "마지막에는 내일에 대한 다짐이나 소망을 한두 문장 덧붙이세요."
        )
    else:
        sys_prompt = (
            "당신은 감정 분석에 기반한 한국어 독후감 작성 도우미입니다. "
            "사용자의 답변과 감정을 참고하여 작품의 줄거리, 주제, 인상 깊었던 장면, "
            "느낀 점과 배운 점을 서론-본론-결론 구조로 정리해 주세요. "
            "문체는 자연스럽고 진솔한 1인칭 독후감 형식으로 작성하세요."
        )

    user_prompt = f"다음은 사용자와의 인터뷰 대화 내용입니다:\n\n{qa_block}\n\n위 내용을 바탕으로 글을 작성하세요."

    return openai_chat(MODEL_DEEP, sys_prompt, user_prompt)


# ============================
# FastAPI 앱 생성 (Swagger 경로 Spring 스타일로)
# ============================

app = FastAPI(
    title="AI Writing API",
    version="1.0.0",
    openapi_url="/v3/api-docs",
    docs_url=None,
    redoc_url=None,
)

@app.get("/swagger-ui.html")
async def swagger_ui():
    return get_swagger_ui_html(
        openapi_url="/v3/api-docs",
        title="AI Writing API"
    )

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# 엔드포인트 구현
# ============================

# 1) 세션 시작
@app.post("/api/ai/start", response_model=StartResponse, tags=["ai-controller"])
def ai_start(req: StartRequest):
    if req.mode not in ("diary", "review"):
        raise HTTPException(status_code=400, detail="mode는 'diary' 또는 'review'만 가능합니다.")

    session_id = str(uuid4())
    session = WritingSession(mode=req.mode)
    initial_q = generate_initial_question(req.mode)
    session.current_question = initial_q
    sessions[session_id] = session

    return StartResponse(
        sessionId=session_id,
        question=initial_q,
        step=1,
        targetInitial=TARGET_INITIAL_QA,
    )


# 2) 답변 + 감정 분석
@app.post("/api/ai/answer", response_model=AnswerResponse, tags=["ai-controller"])
def ai_answer(req: AnswerRequest):
    session = sessions.get(req.sessionId)
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    if not session.current_question:
        raise HTTPException(status_code=400, detail="현재 답변할 질문이 없습니다. 흐름을 다시 확인해주세요.")

    # 1) 감정 분석
    emo = emotion_model.predict(req.answer)

    # 2) QA 저장 (초기/추가 질문 구분은 is_extra 플래그로만 기록)
    is_extra = session.finished_initial  # 초기 5개 끝난 이후부터는 전부 extra
    qa = QAPair(
        question=session.current_question,
        answer=req.answer,
        emotion=emo["emotion"],
        probs=emo["probs"],
        is_extra=is_extra,
    )
    session.qa.append(qa)

    # 3) 좋은 질문 피드백
    if req.likeQuestion is True:
        save_good_question(session.current_question, session.mode)

    total_qa = len(session.qa)

    # --------------------------
    # (A) 초기 5개 질문 단계
    # --------------------------
    if not session.finished_initial:
        if total_qa >= TARGET_INITIAL_QA:
            # 방금 답변으로 초기 질문 5개 채워짐
            session.finished_initial = True
            session.current_question = None
            # 이 시점부터는 /api/ai/result 로 초안 생성 가능
            return AnswerResponse(
                sessionId=req.sessionId,
                emotion=emo["emotion"],
                probs=emo["probs"],
                totalQA=total_qa,
                needMoreQA=False,
                canGenerateResult=True,
                nextQuestion=None,
            )
        else:
            # 아직 초기 질문 단계 계속
            next_q = generate_next_question(session, is_extra=False)
            session.current_question = next_q
            return AnswerResponse(
                sessionId=req.sessionId,
                emotion=emo["emotion"],
                probs=emo["probs"],
                totalQA=total_qa,
                needMoreQA=True,
                canGenerateResult=False,
                nextQuestion=next_q,
            )

    # --------------------------
    # (B) 추가 질문 단계 (feedback에서 addN 설정 후)
    # --------------------------
    if session.extra_target > 0:
        session.extra_done += 1

        # 아직 addN만큼 다 못 채움 → 다음 추가 질문 필요
        if session.extra_done < session.extra_target:
            next_q = generate_next_question(session, is_extra=True)
            session.current_question = next_q
            return AnswerResponse(
                sessionId=req.sessionId,
                emotion=emo["emotion"],
                probs=emo["probs"],
                totalQA=total_qa,
                needMoreQA=True,          # 아직 더 물어봐야 함
                canGenerateResult=False,   # 아직 결과 재생성 단계 아님
                nextQuestion=next_q,
            )
        else:
            # addN 만큼 추가 질문 모두 완료
            session.current_question = None
            session.extra_target = 0
            session.extra_done = 0
            return AnswerResponse(
                sessionId=req.sessionId,
                emotion=emo["emotion"],
                probs=emo["probs"],
                totalQA=total_qa,
                needMoreQA=False,
                canGenerateResult=True,    # 다시 /api/ai/result 호출해서 재작성 가능
                nextQuestion=None,
            )

    # --------------------------
    # (C) 초기도 끝났고 extra_target도 없는 경우
    # --------------------------
    session.current_question = None
    return AnswerResponse(
        sessionId=req.sessionId,
        emotion=emo["emotion"],
        probs=emo["probs"],
        totalQA=total_qa,
        needMoreQA=False,
        canGenerateResult=True,
        nextQuestion=None,
    )



# 4) 만족도 피드백
@app.post("/api/ai/feedback", response_model=FeedbackResponse, tags=["ai-controller"])
def ai_feedback(req: FeedbackRequest):
    session = sessions.get(req.sessionId)
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    # 사용자가 만족한 경우 → 세션 종료
    if req.satisfied:
        session.satisfied = True
        session.extra_target = 0
        session.extra_done = 0
        session.current_question = None
        return FeedbackResponse(
            sessionId=req.sessionId,
            status="done",
            question=None,
            totalQA=len(session.qa),
        )

    # 만족하지 않은 경우 → addN 만큼 추가 질문 루프 진입
    add_n = req.addN if req.addN and req.addN > 0 else 1
    session.satisfied = False
    session.extra_target = add_n
    session.extra_done = 0

    # 첫 번째 추가 질문 생성
    extra_q = generate_next_question(session, is_extra=True)
    session.current_question = extra_q

    return FeedbackResponse(
        sessionId=req.sessionId,
        status="need_more_qa",
        question=extra_q,
        totalQA=len(session.qa),
    )



# 5) 세션 완료 마킹
@app.post("/api/ai/{sessionId}/complete", response_model=FeedbackResponse, tags=["ai-controller"])
def ai_complete(sessionId: str):
    session = sessions.get(sessionId)
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    session.satisfied = True
    return FeedbackResponse(
        sessionId=sessionId,
        status="done",
        question=None,
        totalQA=len(session.qa),
    )


# ============================
# 실행
# ============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai_main_code_server:app", host="0.0.0.0", port=60013, reload=True)
