# emotion_predict.py
# KoBERT 감정 분석 예측 코드 (서버/CLI 공용)
import os
import sys

from emotion_model import EmotionPredictor


predictor = EmotionPredictor(
    model_path=os.getenv("EMOTION_MODEL_PATH", "models/emotion_model.pt"),
    classes_path=os.getenv("EMOTION_CLASSES_PATH", "models/classes.npy"),
)


def predict_emotion(text: str):
    result = predictor.predict(text)
    probs = dict(
        zip(
            predictor.classes.tolist(),
            [round(p, 3) for p in result["probs"]],
        )
    )
    return result["emotion"], probs


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/emotion_predict.py "문장 내용"')
        sys.exit(1)

    text = sys.argv[1]
    emotion, probs = predict_emotion(text)
    print(f"예측 감정: {emotion}")
    print(f"감정 확률: {probs}")
