# emotion_predict.py
# KoBERT 감정 분석 예측 코드 (서버용)
import sys
import torch
import torch.nn as nn
import numpy as np
from kobert_transformers import get_kobert_model, get_tokenizer

# 모델 구조 동일하게 정의
class EmotionClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = get_kobert_model()
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs[1]
        out = self.dropout(pooled_output)
        return self.fc(out)

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = np.load("classes.npy", allow_pickle=True)
model = EmotionClassifier(num_classes=len(classes))
model.load_state_dict(torch.load("kobert_emotion_best.pt", map_location=device))
model.to(device)
model.eval()

# 감정 예측 함수
def predict_emotion(text):
    tokenizer = get_tokenizer()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred = np.argmax(probs)
    return classes[pred], dict(zip(classes, np.round(probs, 3)))

# CLI 입력 처리 (Node.js 연동)
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python emotion_predict.py \"문장 내용\"")
        sys.exit(1)

    text = sys.argv[1]
    emotion, probs = predict_emotion(text)
    print(f"예측 감정: {emotion}")
    print(f"감정 확률: {probs}")
