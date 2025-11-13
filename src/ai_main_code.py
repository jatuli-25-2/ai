"""
AI 메인 프로그램 (감정 기반 일기/독후감 생성)
GPT-4o-mini: 질문 생성 / 음악 추천
GPT-4o: 본문 생성
"""

import os
import argparse
from datetime import datetime
from typing_extensions import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BertModel
from openai import OpenAI
from langchain.memory import ConversationBufferMemory

# 경로 설정
os.chdir(os.path.dirname(__file__))

# 환경 설정
os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxx"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_FAST = "gpt-4o-mini"
MODEL_DEEP = "gpt-4o"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMOTIONS_KO = ["분노", "혐오", "두려움", "기쁨", "중립", "슬픔", "놀람"]

# 학습된 감정분석모델 사용
class EmotionClassifier7(nn.Module):
    """학습 완료된 KoBERT 기반 감정 분류기"""
    def __init__(self, model_path="emotion_tuning.pt"):
        super().__init__()
        self.bert = BertModel.from_pretrained("monologg/kobert", trust_remote_code=True)
        self.fc = nn.Linear(self.bert.config.hidden_size, 7)

        state_dict = torch.load(model_path, map_location=DEVICE)
        self.load_state_dict(state_dict, strict=False)

        self.to(DEVICE)
        self.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)

    @torch.no_grad()
    def predict(self, text: str) -> Dict[str, Any]:
        """입력 문장의 감정 및 확률 분포 반환"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64
        ).to(DEVICE)
        logits = self.fc(self.bert(**inputs).last_hidden_state[:, 0])
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        idx = int(probs.argmax())
        return {"emotion": EMOTIONS_KO[idx], "probs": probs.tolist()}

# 세션 관리
class WritingSession:
    def __init__(self, mode: str, emo_model: EmotionClassifier7):
        self.mode = mode
        self.emo_model = emo_model
        self.memory = ConversationBufferMemory(return_messages=True)
        self.qa_pairs = {}
        self.emotions = {}

    def save_output(self, text: str):
        os.makedirs("outputs", exist_ok=True)
        filename = f"outputs/{self.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n결과 저장 완료 : {filename}")


# OpenAI Chat 호출
def openai_chat(model: str, sys: str, user: str, max_tokens=400):
    """OpenAI Chat 호출"""
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI API 호출 실패:", e)
        return "죄송합니다. 잠시 후 다시 시도해주세요."

# 감정 기반 음악 추천
def recommend_music(emotion: str) -> Dict[str, str]:
    """감정 기반 음악 추천"""
    prompt = f"한국 노래 중 '{emotion}' 감정에 어울리는 노래 1곡과 유튜브 링크를 추천해줘."
    res = openai_chat(MODEL_FAST, "너는 한국 음악 큐레이터야.", prompt)
    return {"emotion": emotion, "recommendation": res}


# 메인 로직
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["diary", "review"], required=True)
    parser.add_argument("--emotion_model_path", default="emotion_tuning.pt")
    args = parser.parse_args()

    emo_clf = EmotionClassifier7(model_path=args.emotion_model_path)
    session = WritingSession(args.mode, emo_clf)

    # 모드별 첫 질문
    question = (
        "오늘 하루 어땠나요?" if args.mode == "diary"
        else "이 책을 읽게 된 계기나 이유는 무엇인가요?"
    )

# 기본 질의응답 진행(5회)
    for i in range(5):
        print(f"\nQ{i+1}. {question}")
        ans = input("A: ")

        emo = session.emo_model.predict(ans)
        print(f"💬 감정: {emo['emotion']} | 확률분포: {['%.2f' % p for p in emo['probs']]}")

        session.qa_pairs[question] = ans
        session.emotions[question] = emo["emotion"]

        if i < 4:
            # 대화 이력 및 감정분석 기반 질문 생성
            qa_history = "\n".join(
                [f"Q: {q}\nA: {a} (감정: {session.emotions[q]})" for q, a in session.qa_pairs.items()]
            )
            emotion_trend = list(session.emotions.values())

            next_q_prompt = f"""
            지금까지의 대화는 다음과 같습니다:
            {qa_history}

            사용자의 감정 변화 흐름: {emotion_trend}

            위 대화 내용을 바탕으로,
            - 이미 다룬 주제나 질문을 반복하지 말고,
            - 새로운 감정적 측면이나 구체적 상황을 탐색하며,
            - 다음에 이어질 자연스럽고 진정성 있는 질문 한 개를 제시하세요.
            (단, 너무 추상적이거나 일반적인 질문은 피하고 구체적인 상황 중심으로 질문하세요.)
            """

            next_q = openai_chat(
                MODEL_FAST,
                "당신은 감정의 흐름을 파악하며 대화하는 한국어 인터뷰어입니다.",
                next_q_prompt
            )

            question = next_q


    # 모드별 프롬프트
    if args.mode == "diary":
        sys_prompt = (
            "당신은 감정에 공감하는 한국어 일기 작성 도우미입니다. "
            "사용자의 감정과 하루의 흐름을 반영해, 따뜻하고 진솔한 일기를 작성하세요. "
            "문체는 자연스러운 1인칭 시점이며, 마지막에는 내일의 다짐이나 소망을 덧붙이세요."
        )
    else:
        sys_prompt = (
            "당신은 감정 분석에 기반한 한국어 독후감 작성 도우미입니다. "
            "사용자의 감정과 답변을 참고하여, 작품의 주제·느낌·배운 점을 중심으로 "
            "논리적이지만 감정이 살아있는 독후감을 작성하세요. "
            "형식은 서론-본론-결론 구조를 따르며, 마크다운 형식을 활용하세요."
        )

    # 사용자 프롬프트 구성
    user_prompt = "\n".join(
        [f"Q: {q}\nA: {a} (감정: {session.emotions[q]})" for q, a in session.qa_pairs.items()]
    )

    # 결과물 생성
    final_text = openai_chat(MODEL_DEEP, sys_prompt, user_prompt)
    print("\n📝 생성된 결과물:\n")
    print(final_text)

    # 만족도 평가 및 추가 Q&A
    feedback = input("\n결과물이 마음에 드시나요? (y/n): ").strip().lower()
    if feedback == "n":
        add_n = int(input("몇 개의 추가 질문을 진행할까요? (1~3): "))
        for j in range(add_n):
            question = openai_chat(
                MODEL_FAST,
                "당신은 감정 기반 인터뷰어입니다.",
                f"모드: {args.mode}, 기존 감정 리스트: {list(session.emotions.values())}. "
                f"추가로 깊이 있는 대화를 위한 질문 하나를 제시하세요."
            )
            print(f"\n추가 Q{j+1}. {question}")
            ans = input("A: ")

            emo = session.emo_model.predict(ans)
            print(f"💬 감정: {emo['emotion']} | 확률분포: {['%.2f' % p for p in emo['probs']]}")

            session.qa_pairs[question] = ans
            session.emotions[question] = emo["emotion"]

        # 결과물 재생성
        user_prompt = "\n".join(
            [f"Q: {q}\nA: {a} (감정: {session.emotions[q]})" for q, a in session.qa_pairs.items()]
        )
        final_text = openai_chat(MODEL_DEEP, sys_prompt, user_prompt)
        print("\n수정된 결과물:\n")
        print(final_text)

    # 감정 기반 음악 추천
    dominant_emotion = max(
        session.emotions.values(),
        key=lambda e: list(session.emotions.values()).count(e)
    )
    music = recommend_music(dominant_emotion)
    print("\n감정 기반 음악 추천:")
    print(music["recommendation"])

    # 결과 저장
    session.save_output(final_text)


if __name__ == "__main__":
    main()
