"""Shared KoBERT emotion classification model used by training and serving."""

import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from kobert_transformers import get_kobert_model, get_tokenizer

EMOTION_KO = {
    "angry": "분노",
    "disgust": "혐오",
    "fear": "두려움",
    "happiness": "기쁨",
    "neutral": "중립",
    "sadness": "슬픔",
    "surprise": "놀람",
    "분노": "분노",
    "혐오": "혐오",
    "두려움": "두려움",
    "기쁨": "기쁨",
    "중립": "중립",
    "슬픔": "슬픔",
    "놀람": "놀람",
}


class EmotionClassifier(nn.Module):
    """KoBERT + Dropout + linear head used for 7-class emotion classification."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.bert = get_kobert_model()
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = outputs[1]
        out = self.dropout(pooled_output)
        return self.fc(out)


class EmotionPredictor:
    """Load the trained classifier and class mapping, then expose text prediction."""

    def __init__(
        self,
        model_path: str = "models/emotion_model.pt",
        classes_path: str = "models/classes.npy",
        device: Optional[torch.device] = None,
        max_length: int = 64,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        # Environment variables take priority in deployment environments.
        self.model_path = os.getenv("EMOTION_MODEL_PATH", model_path)
        self.classes_path = os.getenv("EMOTION_CLASSES_PATH", classes_path)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                "감정분류 모델 파일을 찾을 수 없습니다: "
                f"{self.model_path}. models/README.md를 참고해 모델 파일을 배치하세요."
            )

        if not os.path.exists(self.classes_path):
            raise FileNotFoundError(
                "감정분류 클래스 매핑 파일을 찾을 수 없습니다: "
                f"{self.classes_path}. 학습 시 생성된 classes.npy를 모델과 함께 배치하세요."
            )

        self.classes = np.load(self.classes_path, allow_pickle=True)

        self.model = EmotionClassifier(num_classes=len(self.classes))
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = get_tokenizer()

    @torch.no_grad()
    def predict(self, text: str) -> Dict[str, Any]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        ).to(self.device)

        logits = self.model(**inputs)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        raw_emotion = str(self.classes[idx])

        return {
            "emotion": EMOTION_KO.get(raw_emotion, raw_emotion),
            "raw_emotion": raw_emotion,
            "probs": probs.tolist(),
        }
