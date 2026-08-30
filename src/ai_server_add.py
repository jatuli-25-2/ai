"""Final stateless FastAPI AI server for Jatuli.

Spring owns the conversation history and sends the complete message list on each request.
The AI server combines KoBERT emotion recognition with LLM-based question generation,
writing, recommendations, and title suggestions.
"""

import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel

from emotion_model import EmotionPredictor


# ============================
# Environment
# ============================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

client = OpenAI(api_key=OPENAI_API_KEY)

# Fast/high-volume tasks and deeper final writing can be configured independently.
MODEL_FAST = os.getenv("OPENAI_MODEL_FAST", "gpt-5.6-luna")
MODEL_DEEP = os.getenv("OPENAI_MODEL_DEEP", "gpt-5.6-terra")

MUSIC_GENRES = ["발라드", "댄스", "힙합", "R&B", "인디", "록", "OST", "트로트", "기타"]
BOOK_GENRES = ["소설", "에세이", "자기계발", "인문", "심리", "시", "기타"]

emotion_model = EmotionPredictor(
    model_path="emotion_model.pt",
    classes_path="classes.npy",
)

app = FastAPI(
    title="Jatuli AI Server",
    version="2.0.0",
    description="KoBERT emotion recognition + LLM writing assistant",
)


# ============================
# Pydantic models
# ============================
class MessageItem(BaseModel):
    role: str  # "AI" or "USER"
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
    titles: Optional[List[str]] = None
    finalTitle: Optional[str] = None
    allowCustom: bool = True
    stage: str  # "suggest" or "confirm"


# ============================
# Common helpers
# ============================
def openai_chat(model: str, system: str, user: str, max_tokens: int = 400) -> str:
    res = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_completion_tokens=max_tokens,
    )
    content = res.choices[0].message.content
    return (content or "").strip()


def _clip_for_prompt(text: str, max_chars: int = 1600) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:1200] + "\n...\n" + text[-400:]


def _extract_json_loose(text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction for models that add text around a JSON object."""
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def openai_json(model: str, system: str, user: str, max_tokens: int = 250) -> Dict[str, Any]:
    """Request JSON output and fall back to loose extraction when necessary."""
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_completion_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        content = res.choices[0].message.content or ""
        return json.loads(content.strip())
    except Exception:
        raw = openai_chat(model, system, user, max_tokens=max_tokens)
        return _extract_json_loose(raw)


def analyze_user_messages(messages: List[MessageItem]) -> List[Dict[str, str]]:
    """Return user messages paired with KoBERT emotion predictions."""
    analyzed: List[Dict[str, str]] = []
    for message in messages:
        if message.role.upper() != "USER":
            continue
        prediction = emotion_model.predict(message.content)
        analyzed.append(
            {
                "content": message.content,
                "emotion": str(prediction["emotion"]),
            }
        )
    return analyzed


def format_emotion_context(analyzed: List[Dict[str, str]]) -> str:
    if not analyzed:
        return "분석된 사용자 발화가 없습니다."
    return "\n".join(
        f"USER: {item['content']}\n감정 분석: {item['emotion']}"
        for item in analyzed
    )


# ============================
# Search-link helpers
# ============================
def youtube_search_url(title: str, artist: str) -> str:
    query = quote_plus(f"{title} {artist}".strip())
    return f"https://www.youtube.com/results?search_query={query}"


def kyobo_search_url(title: str, author: str) -> str:
    query = quote_plus(f"{title} {author}".strip())
    return f"https://search.kyobobook.co.kr/search?keyword={query}"


def aladin_search_url(title: str, author: str) -> str:
    query = quote_plus(f"{title} {author}".strip())
    return f"https://www.aladin.co.kr/search/wsearchresult.aspx?SearchTarget=All&SearchWord={query}"


# ============================
# Recommendations
# ============================
def recommend_music(emotion: str) -> Dict[str, str]:
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

    if not title or not artist:
        title, artist = "추천 곡을 불러오지 못했습니다", ""
        reason, genre = "잠시 후 다시 시도해주세요.", "기타"

    link = youtube_search_url(title, artist) if title and artist else ""
    rec_text = (
        f"🎵 추천: {title} - {artist}\n"
        f"장르: {genre}\n"
        f"이유: {reason}\n"
        f"유튜브에서 찾아보기: {link}"
    )
    return {"type": genre, "emotion": emotion, "recommend": rec_text}


def recommend_book(emotion: str) -> Dict[str, str]:
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

    if not title or not author:
        title, author = "추천 도서를 불러오지 못했습니다", ""
        one_line, reason, genre = "", "잠시 후 다시 시도해주세요.", "기타"

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
# Title suggestion
# ============================
def suggest_titles(mode: str, final_text: str, dominant_emotion: Optional[str] = None) -> List[str]:
    clipped = _clip_for_prompt(final_text)
    style_hint = (
        "따뜻하고 감성적인 일기 제목"
        if mode == "diary"
        else "핵심 주제와 통찰이 드러나는 독후감 제목"
    )

    prompt = f"""
아래 글에 어울리는 {style_hint} 3개를 추천해줘.
조건:
- 한국어
- 10~25자 정도
- 서로 겹치지 않게 표현과 키워드를 다양화
- 따옴표/이모지/번호/불릿 없이 제목 문장만 작성
- 반드시 다음 JSON 형식으로만 출력
{{"titles":["...","...","..."]}}

참고 감정: {dominant_emotion or "없음"}

[글]
{clipped}
""".strip()

    data = openai_json(
        MODEL_FAST,
        "너는 한국어 글 제목을 잘 뽑는 에디터다. 반드시 JSON만 출력한다.",
        prompt,
        max_tokens=200,
    )

    titles = data.get("titles", []) if isinstance(data, dict) else []
    titles = [str(title).strip()[:30] for title in titles if str(title).strip()]
    titles = list(dict.fromkeys(titles))[:3]

    while len(titles) < 3:
        fallback = (
            f"오늘의 기록 {len(titles) + 1}"
            if mode == "diary"
            else f"읽고 남은 생각 {len(titles) + 1}"
        )
        titles.append(fallback)

    return titles


# ============================
# API endpoints
# ============================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "jatuli-ai",
        "modelFast": MODEL_FAST,
        "modelDeep": MODEL_DEEP,
    }


@app.get("/api/ai/start")
def get_first_question(mode: str):
    if mode == "diary":
        question = "오늘 하루 중 가장 기억에 남는 순간은 무엇이었나요?"
    else:
        question = "최근 읽은 책은 무엇이며, 선택한 이유는 무엇인가요?"
    return {"question": question}


@app.post("/api/ai/next-question", response_model=NextQuestionResponse)
def next_question(req: NextQuestionRequest):
    history = "\n".join(f"{m.role}: {m.content}" for m in req.messages)
    analyzed = analyze_user_messages(req.messages)
    latest_emotion = analyzed[-1]["emotion"] if analyzed else "중립"

    prompt = f"""
다음은 사용자와 AI의 대화입니다.

[대화 기록]
{history}

[최근 사용자 감정]
{latest_emotion}

위 대화와 최근 감정을 함께 고려해 다음 질문을 생성하세요.
- 이미 했던 질문을 반복하지 마세요.
- 사용자의 최근 답변과 감정에 자연스럽게 이어지게 하세요.
- 감정을 단정하거나 진단하지 말고, 사용자가 더 이야기할 수 있도록 질문하세요.
- 질문은 반드시 한국어 한 문장만 출력하세요.
""".strip()

    next_q = openai_chat(
        MODEL_FAST,
        "너는 사용자의 대화 흐름과 감정을 참고해 후속 질문을 만드는 한국어 인터뷰어입니다.",
        prompt,
    )

    return NextQuestionResponse(nextQuestion=next_q, emotion=latest_emotion)


@app.post("/api/ai/finalize", response_model=FinalizeResponse)
def finalize(req: FinalizeRequest):
    history = "\n".join(f"{m.role}: {m.content}" for m in req.messages)
    analyzed = analyze_user_messages(req.messages)
    emotions = [item["emotion"] for item in analyzed]
    dominant_emotion = Counter(emotions).most_common(1)[0][0] if emotions else "중립"
    emotion_context = format_emotion_context(analyzed)

    if req.mode == "diary":
        system_prompt = (
            "당신은 감정 기반 한국어 일기 작성 어시스턴트입니다. "
            "사용자의 실제 발화 내용이 중심이 되도록 하고, 감정 분석 결과는 문맥을 이해하기 위한 보조 정보로만 활용하세요. "
            "사용자가 말하지 않은 사건이나 감정을 임의로 만들어내지 말고 따뜻하고 자연스러운 1인칭 일기를 작성하세요."
        )
    else:
        system_prompt = (
            "당신은 감정 기반 한국어 독후감 작성 어시스턴트입니다. "
            "사용자의 실제 발화 내용이 중심이 되도록 하고, 감정 분석 결과는 문맥을 이해하기 위한 보조 정보로만 활용하세요. "
            "사용자가 말하지 않은 내용을 임의로 만들어내지 말고 서론-본론-결론 구조의 자연스러운 1인칭 독후감을 작성하세요."
        )

    generation_prompt = f"""
[전체 대화]
{history}

[사용자 발화별 감정 분석]
{emotion_context}

[대표 감정]
{dominant_emotion}

위 정보를 바탕으로 최종 글을 작성하세요.
""".strip()

    final_text = openai_chat(MODEL_DEEP, system_prompt, generation_prompt, max_tokens=800)
    rec_obj = (
        recommend_music(dominant_emotion)
        if req.mode == "diary"
        else recommend_book(dominant_emotion)
    )

    return FinalizeResponse(
        finalText=final_text,
        dominantEmotion=dominant_emotion,
        recommend=rec_obj,
    )


@app.post("/api/ai/title", response_model=TitleResponse)
def title(req: TitleRequest):
    if req.customTitle and req.customTitle.strip():
        return TitleResponse(
            finalTitle=req.customTitle.strip()[:50],
            allowCustom=True,
            stage="confirm",
        )

    if req.selectedIndex is not None:
        titles = req.titles or suggest_titles(req.mode, req.finalText, req.dominantEmotion)
        if 0 <= req.selectedIndex < len(titles):
            chosen = (titles[req.selectedIndex] or "").strip()
            if chosen:
                return TitleResponse(
                    finalTitle=chosen[:50],
                    allowCustom=True,
                    stage="confirm",
                )
        return TitleResponse(
            finalTitle="제목 없음",
            allowCustom=True,
            stage="confirm",
        )

    titles = suggest_titles(req.mode, req.finalText, req.dominantEmotion)
    return TitleResponse(
        titles=titles,
        allowCustom=True,
        stage="suggest",
    )
