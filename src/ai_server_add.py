# ai/src/ai_server.py
# ⛔ stateful 없음
# ⭕ Spring에서 보낸 messages 전체 기반으로 항상 동작하는 순수 생성기

import os
import json
import re
from urllib.parse import quote_plus
from typing import List, Dict, Optional, Any
from collections import Counter

from fastapi import FastAPI
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

MODEL_FAST = "gpt-5.2-chat-latest"
MODEL_DEEP = "gpt-5.2"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTIONS_KO = ["분노", "혐오", "두려움", "기쁨", "중립", "슬픔", "놀람"]

# 추천 장르
MUSIC_GENRES = ["발라드", "댄스", "힙합", "R&B", "인디", "록", "OST", "트로트", "기타"]
BOOK_GENRES = ["소설", "에세이", "자기계발", "인문", "심리", "시", "기타"]

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
        return {"emotion": EMOTIONS_KO[idx], "probs": probs.tolist()}

emotion_model = EmotionClassifier7(model_path="emotion_model.pt")

# ============================
# Pydantic Models
# ============================
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
    recommend: dict

class TitleRequest(BaseModel):
    mode: str
    finalText: str
    dominantEmotion: Optional[str] = None

    titles: Optional[List[str]] = None
    selectedIndex: Optional[int] = None
    customTitle: Optional[str] = None

class TitleResponse(BaseModel):
    titles: Optional[List[str]] = None     # 제안 후보 3개
    finalTitle: Optional[str] = None       # 확정 제목 1개
    allowCustom: bool = True
    stage: str  # "suggest" or "confirm"

app = FastAPI()

# ============================
# OpenAI 공통 함수
# ============================
def openai_chat(model: str, system: str, user: str, max_tokens: int = 400) -> str:
    res = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_completion_tokens=max_tokens,
        temperature=0.7,
    )
    return res.choices[0].message.content.strip()

def _clip_for_prompt(text: str, max_chars: int = 1600) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:1200] + "\n...\n" + text[-400:]

# ============================
# JSON 안전 파싱 유틸 (추천/제목에서 사용)
# ============================
def _extract_json_loose(text: str) -> Dict[str, Any]:
    """
    모델이 JSON만 주기로 했는데 앞뒤 텍스트를 붙이는 경우가 있어
    { ... } 블록만 최대한 뽑아 파싱한다.
    """
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}

def openai_json(model: str, system: str, user: str, max_tokens: int = 250) -> Dict[str, Any]:
    """
    JSON 전용 호출.
    response_format 지원되면 안정적으로 JSON만 받음.
    미지원/실패 시 loose 파싱 fallback.
    """
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_completion_tokens=max_tokens,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return json.loads(res.choices[0].message.content.strip())
    except Exception:
        raw = openai_chat(model, system, user, max_tokens=max_tokens)
        return _extract_json_loose(raw)

# ============================
# 링크 생성 유틸
# ============================
def youtube_search_url(title: str, artist: str) -> str:
    q = quote_plus(f"{title} {artist}".strip())
    return f"https://www.youtube.com/results?search_query={q}"

def kyobo_search_url(title: str, author: str) -> str:
    q = quote_plus(f"{title} {author}".strip())
    return f"https://search.kyobobook.co.kr/search?keyword={q}"

def aladin_search_url(title: str, author: str) -> str:
    q = quote_plus(f"{title} {author}".strip())
    return f"https://www.aladin.co.kr/search/wsearchresult.aspx?SearchTarget=All&SearchWord={q}"

# ============================
# 추천 유틸 (음악 / 도서)
# ============================
def recommend_music(emotion: str) -> Dict[str, str]:
    """
    - LLM이 URL을 만들지 않음(링크 환각 방지)
    - title/artist/genre/reason을 JSON으로만 받음
    - 링크는 서버가 '검색 링크'로 생성
    """
    prompt = f"""
아래 형식의 JSON만 출력하세요. 다른 텍스트/링크/마크다운 금지.

필드:
- title: 한국 노래 제목(실존/대중적으로 알려진 곡)
- artist: 가수명
- genre: {MUSIC_GENRES} 중 1개
- reason: '{emotion}' 감정에 어울리는 이유 (한국어 2~3문장)

규칙:
- URL/링크는 절대 포함하지 마세요.
- 존재가 불확실한 곡은 피하고, 최대한 유명한 곡을 선택하세요.

출력 예:
{{"title":"...","artist":"...","genre":"...","reason":"..."}}
""".strip()

    data = openai_json(MODEL_FAST, "너는 한국 음악 큐레이터야.", prompt, max_tokens=220)

    title = str(data.get("title", "")).strip()
    artist = str(data.get("artist", "")).strip()
    genre = str(data.get("genre", "기타")).strip()
    reason = str(data.get("reason", "")).strip()

    if genre not in MUSIC_GENRES:
        genre = "기타"

    # 안전장치
    if not title or not artist:
        title, artist, reason, genre = "추천 곡을 불러오지 못했습니다", "", "잠시 후 다시 시도해주세요.", "기타"

    link = youtube_search_url(title, artist) if title and artist else ""

    rec_text = (
        f"🎵 추천: {title} - {artist}\n"
        f"장르: {genre}\n"
        f"이유: {reason}\n"
        f"유튜브에서 찾아보기: {link}"
    )

    return {"type": genre, "emotion": emotion, "recommend": rec_text}

def recommend_book(emotion: str) -> Dict[str, str]:
    """
    - LLM이 서점 URL을 만들지 않음(링크 환각 방지)
    - title/author/genre/one_line/reason을 JSON으로만 받음
    - 링크는 서버가 '검색 링크'로 생성
    """
    prompt = f"""
아래 형식의 JSON만 출력하세요. 다른 텍스트/링크/마크다운 금지.

필드:
- title: 한국 도서 제목(실존 도서)
- author: 저자
- genre: {BOOK_GENRES} 중 1개
- one_line: 한 줄 줄거리(한국어 1문장)
- reason: '{emotion}' 감정에 어울리는 이유 (한국어 2~3문장)

규칙:
- URL/링크는 절대 포함하지 마세요.
- 존재가 불확실한 도서는 피하고, 최대한 검증된 도서를 선택하세요.

출력 예:
{{"title":"...","author":"...","genre":"...","one_line":"...","reason":"..."}}
""".strip()

    data = openai_json(MODEL_FAST, "너는 한국 도서 큐레이터야.", prompt, max_tokens=260)

    title = str(data.get("title", "")).strip()
    author = str(data.get("author", "")).strip()
    genre = str(data.get("genre", "기타")).strip()
    one_line = str(data.get("one_line", "")).strip()
    reason = str(data.get("reason", "")).strip()

    if genre not in BOOK_GENRES:
        genre = "기타"

    # 안전장치
    if not title or not author:
        title, author, one_line, reason, genre = "추천 도서를 불러오지 못했습니다", "", "", "잠시 후 다시 시도해주세요.", "기타"

    kyobo = kyobo_search_url(title, author) if title and author else ""
    aladin = aladin_search_url(title, author) if title and author else ""

    rec_text = (
        f"📚 추천: {title} - {author}\n"
        f"장르: {genre}\n"
        f"한 줄: {one_line}\n"
        f"이유: {reason}\n"
        f"교보문고에서 찾아보기: {kyobo}\n"
        f"알라딘에서 찾아보기: {aladin}"
    )

    return {"type": genre, "emotion": emotion, "recommend": rec_text}

# ============================
# 제목 추천
# ============================
def suggest_titles(mode: str, final_text: str, dominant_emotion: Optional[str] = None) -> List[str]:
    clipped = _clip_for_prompt(final_text)

    system = (
        "너는 한국어 글 제목을 잘 뽑는 에디터다. "
        "반드시 JSON만 출력한다. 다른 말 금지."
    )

    style_hint = (
        "따뜻하고 감성적인 일기 제목"
        if mode == "diary"
        else "핵심 주제와 통찰이 드러나는 독후감 제목"
    )

    user = f"""
아래 글에 어울리는 {style_hint} 3개를 추천해줘.
조건:
- 한국어
- 10~25자 정도
- 서로 겹치지 않게(표현/키워드 다양화)
- 따옴표/이모지/번호/불릿 없이 '제목 문장'만
- 반드시 다음 JSON 형식으로만 출력:
{{"titles":["...","...","..."]}}

참고 감정(있으면 반영): {dominant_emotion or "없음"}

[글]
{clipped}
""".strip()

    raw = openai_chat(MODEL_FAST, system, user, max_tokens=200)

    titles: List[str] = []
    try:
        data = json.loads(raw)
        titles = data.get("titles", []) if isinstance(data, dict) else []
    except Exception:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        cleaned = []
        for ln in lines:
            ln = re.sub(r'^[\-\*\d\.\)\(]+\s*', '', ln).strip()
            ln = ln.strip('"\'')
            if ln:
                cleaned.append(ln)
        titles = cleaned

    titles = [t.strip() for t in titles if isinstance(t, str) and t.strip()]
    titles = [t[:30] for t in titles]
    titles = list(dict.fromkeys(titles))
    titles = titles[:3]

    while len(titles) < 3:
        if mode == "diary":
            titles.append(f"오늘의 기록 {len(titles)+1}")
        else:
            titles.append(f"읽고 남은 생각 {len(titles)+1}")

    return titles

# ============================
# 1) 첫 질문
# ============================
@app.get("/api/ai/start")
def get_first_question(mode: str):
    if mode == "diary":
        question = "오늘 하루 중 가장 기억에 남는 순간은 무엇이었나요?"
    else:
        question = "최근 읽은 책은 무엇이며, 선택한 이유는 무엇인가요?"
    return {"question": question}

# ============================
# 2) 다음 질문 생성 + 마지막 감정 분석
# ============================
@app.post("/api/ai/next-question", response_model=NextQuestionResponse)
def next_question(req: NextQuestionRequest):
    history = "\n".join([f"{m.role}: {m.content}" for m in req.messages])

    prompt = f"""
다음은 사용자와 AI의 대화입니다:

{history}

위 대화를 기반으로,
- 이미 했던 질문을 반복하지 말고
- 자연스럽게 이어질 다음 질문 1개만 생성하세요.
반드시 한국어 한 문장으로만 답하세요.
""".strip()

    next_q = openai_chat(MODEL_FAST, "너는 감정 기반 한국어 인터뷰어입니다.", prompt)

    user_messages = [m.content for m in req.messages if m.role.upper() == "USER"]
    if user_messages:
        last_answer = user_messages[-1]
        emo = emotion_model.predict(last_answer)
        emotion_label = emo["emotion"]
    else:
        emotion_label = "중립"

    return NextQuestionResponse(nextQuestion=next_q, emotion=emotion_label)

# ============================
# 3) 최종 글 생성 + 지배적인 감정 + 추천
# ============================
@app.post("/api/ai/finalize", response_model=FinalizeResponse)
def finalize(req: FinalizeRequest):
    history = "\n".join([f"{m.role}: {m.content}" for m in req.messages])

    user_messages = [m.content for m in req.messages if m.role.upper() == "USER"]
    emotions: List[str] = []
    for text in user_messages:
        emo = emotion_model.predict(text)
        emotions.append(emo["emotion"])

    if emotions:
        dominant_emotion = Counter(emotions).most_common(1)[0][0]
    else:
        dominant_emotion = "중립"

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

    if req.mode == "diary":
        rec_obj = recommend_music(dominant_emotion)
    else:
        rec_obj = recommend_book(dominant_emotion)

    return FinalizeResponse(
        finalText=final_text,
        dominantEmotion=dominant_emotion,
        recommend=rec_obj
    )

# ============================
# 4) 제목 추천/확정 (단일 엔드포인트)
# ============================
@app.post("/api/ai/title", response_model=TitleResponse)
def title(req: TitleRequest):
    # 확정 단계: 직접 입력 우선
    if req.customTitle and req.customTitle.strip():
        return TitleResponse(
            finalTitle=req.customTitle.strip()[:50],
            allowCustom=True,
            stage="confirm"
        )

    # 확정 단계: 인덱스 선택
    if req.selectedIndex is not None:
        titles = req.titles or suggest_titles(req.mode, req.finalText, req.dominantEmotion)
        if 0 <= req.selectedIndex < len(titles):
            chosen = (titles[req.selectedIndex] or "").strip()
            if chosen:
                return TitleResponse(
                    finalTitle=chosen[:50],
                    allowCustom=True,
                    stage="confirm"
                )

        return TitleResponse(
            finalTitle="제목 없음",
            allowCustom=True,
            stage="confirm"
        )

    # 제안 단계
    titles = suggest_titles(req.mode, req.finalText, req.dominantEmotion)
    return TitleResponse(
        titles=titles,
        allowCustom=True,
        stage="suggest"
    )
