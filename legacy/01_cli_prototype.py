"""
AI 메인 프로그램 (감정 기반 일기/독후감 생성)
GPT-4o-mini: 질문 생성 / 음악 추천
GPT-4o: 본문 생성
"""

import os
import argparse
import pandas as pd
from datetime import datetime
from typing_extensions import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BertModel
from openai import OpenAI
from langchain.memory import ConversationBufferMemory

# 환경 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_FAST = "gpt-4o-mini"
MODEL_DEEP = "gpt-4o"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMOTIONS_KO = ["분노", "혐오", "두려움", "기쁨", "중립", "슬픔", "놀람"]

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
GOOD_Q_PATH = os.path.join(LOG_DIR, "good_questions.csv")


# 감정분석 진행
class EmotionClassifier7(nn.Module):
    def __init__(self, model_path="emotion_model.pt"):
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
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64).to(DEVICE)
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
        self.log_data = []

    def log_interaction(self, q, a, emo, probs, feedback=None):
        self.log_data.append({
            "질문": q,
            "답변": a,
            "감정": emo,
            "확률분포": probs,
            "사용자피드백": feedback
        })

    def save_logs(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        df = pd.DataFrame(self.log_data)
        df.to_csv(f"{LOG_DIR}/session_log_{self.mode}_{ts}.csv", index=False, encoding="utf-8-sig")
        print(f"세션 로그 저장 완료 → {LOG_DIR}/session_log_{self.mode}_{ts}.csv")

    def save_output(self, text: str):
        os.makedirs("outputs", exist_ok=True)
        filename = f"outputs/{self.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n결과 저장 완료 → {filename}")


# OpenAI 호출
def openai_chat(model: str, sys: str, user: str, max_tokens=400):
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI API 호출 실패:", e)
        return "죄송합니다. 잠시 후 다시 시도해주세요."


# 감정 기반 음악 추천
def recommend_music(emotion: str) -> Dict[str, str]:
    prompt = f"비교적 유명한 한국 노래 중 '{emotion}' 감정에 어울리는 노래 1곡과 유튜브 링크를 추천해줘."
    res = openai_chat(MODEL_FAST, "너는 한국 음악 큐레이터야.", prompt)
    return {"emotion": emotion, "recommendation": res}


# 다음 질문 생성 (예시 포함)
def generate_next_question(mode: str, session: WritingSession, last_emotion: str, last_answer: str, good_examples: list) -> str:
    qa_history = "\n".join([f"Q: {q}\nA: {a} (감정: {session.emotions[q]})" for q, a in session.qa_pairs.items()])
    emotion_trend = list(session.emotions.values())

    # 질문 프롬프트 작성 : 예시 포함
    good_examples_text = "\n".join([f"- {q}" for q in good_examples]) if good_examples else "없음"

    if mode == "diary":
        sys_prompt = "너는 감정을 읽으며 대화를 이끄는 따뜻한 일기 코치입니다."
        user_prompt = f"""
        지금까지의 대화:
        {qa_history}

        최근 감정: {last_emotion}
        최근 답변: {last_answer}
        사용자의 감정 변화 흐름: {emotion_trend}

        사람들이 좋다고 평가한 질문 예시:
        {good_examples_text}

        위 내용을 참고하여,
        - 예시 질문들이나 이전에 했던 질문들을 그대로 반복하지 말고,
        - 감정의 배경이나 이유를 탐색하거나 하루의 긍정적 측면을 이끌어내는 질문을 한 개 만들어주세요.
        """
    elif mode == "book_review":
        sys_prompt = "너는 감정 중심의 독후감 인터뷰어이며, 감정과 통찰을 연결하는 질문을 잘합니다."
        user_prompt = f"""
        지금까지의 대화:
        {qa_history}

        최근 감정: {last_emotion}
        최근 답변: {last_answer}
        감정 흐름: {emotion_trend}

        사람들이 좋다고 평가한 질문 예시:
        {good_examples_text}

        위 내용을 참고하여,
        - 예시 질문이나 이전에 했던 질문들을 단순 복사하지 말고,
        - 작품의 주제나 교훈을 더 깊게 탐구하는 새로운 질문을 만들어주세요.
        """

    return openai_chat(MODEL_FAST, sys_prompt, user_prompt)


# 메인 코드
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["diary", "book_review"], required=True)
    parser.add_argument("--emotion_model_path", default="emotion_model.pt")
    args = parser.parse_args()

    emo_clf = EmotionClassifier7(model_path=args.emotion_model_path)
    session = WritingSession(args.mode, emo_clf)

    # 좋은 질문 불러오기
    prior_good_questions = []
    if os.path.exists(GOOD_Q_PATH):
        df_good = pd.read_csv(GOOD_Q_PATH)
        prior_good_questions = df_good.tail(5)["질문"].tolist()

    question = "오늘 하루 어땠나요?" if args.mode == "diary" else "이 책을 읽게 된 이유는 무엇인가요?"

    # 기본 질의응답 (5회)
    for i in range(5):
        print(f"\nQ{i+1}. {question}")
        ans = input("A: ")

        emo = session.emo_model.predict(ans)
        print(f"감정: {emo['emotion']} | 확률분포: {['%.2f' % p for p in emo['probs']]}")

        session.qa_pairs[question] = ans
        session.emotions[question] = emo["emotion"]
        session.log_interaction(question, ans, emo["emotion"], emo["probs"])

        if i < 4:
            question = generate_next_question(args.mode, session, emo["emotion"], ans, prior_good_questions)

    # 결과물 생성
    sys_prompt = (
        "감정에 공감하는 한국어 일기 작성 도우미입니다. 사용자의 감정과 하루의 흐름을 반영해 따뜻하고 자연스러운 흐름의 1인칭 일기를 작성하세요."
        if args.mode == "diary"
        else "감정과 통찰을 결합한 한국어 독후감 작성 도우미입니다. 서론-본론-결론 구조로 감정이 드러나는 글을 자연스럽게 작성하세요."
    )
    user_prompt = "\n".join([f"Q: {q}\nA: {a} (감정: {session.emotions[q]})" for q, a in session.qa_pairs.items()])
    final_text = openai_chat(MODEL_DEEP, sys_prompt, user_prompt)
    print("\n생성된 결과물:\n")
    print(final_text)

    # 질문 수정 반복
    while True:
        feedback = input("\n결과물이 마음에 드시나요? (y/n): ").strip().lower()
        if feedback == "y":
            break

        add_n = int(input("몇 개의 추가 질문을 진행할까요? (1~3): "))
        for j in range(add_n):
            question = generate_next_question(args.mode, session, list(session.emotions.values())[-1], list(session.qa_pairs.values())[-1], prior_good_questions)
            print(f"\n추가 Q{j+1}. {question}")
            ans = input("A: ")

            emo = session.emo_model.predict(ans)
            print(f"감정: {emo['emotion']} | 확률분포: {['%.2f' % p for p in emo['probs']]}")

            session.qa_pairs[question] = ans
            session.emotions[question] = emo["emotion"]
            session.log_interaction(question, ans, emo["emotion"], emo["probs"])

            good = input("이 질문이 마음에 드셨나요? (y/n): ").strip().lower()
            if good == "y":
                df_append = pd.DataFrame([{"질문": question, "모드": args.mode, "등록일": datetime.now().strftime("%Y-%m-%d")}])
                if os.path.exists(GOOD_Q_PATH):
                    df_append.to_csv(GOOD_Q_PATH, mode="a", header=False, index=False, encoding="utf-8-sig")
                else:
                    df_append.to_csv(GOOD_Q_PATH, index=False, encoding="utf-8-sig")

        # 결과물 재생성
        user_prompt = "\n".join([f"Q: {q}\nA: {a} (감정: {session.emotions[q]})" for q, a in session.qa_pairs.items()])
        final_text = openai_chat(MODEL_DEEP, sys_prompt, user_prompt)
        print("\n수정된 결과물:\n")
        print(final_text)

    # 감정 기반 음악 추천
    dominant_emotion = max(session.emotions.values(), key=lambda e: list(session.emotions.values()).count(e))
    music = recommend_music(dominant_emotion)
    print("\n감정 기반 음악 추천:")
    print(music["recommendation"])

    # 로그 및 결과 저장
    session.save_output(final_text)
    session.save_logs()


if __name__ == "__main__":
    main()
