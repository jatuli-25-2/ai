"""Shared KoBERT emotion classification model used by training and serving."""

import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from kobert_transformers import get_kobert_model, get_tokenizer

# LabelEncoder로 영문 감정 라벨을 정렬했을 때의 순서.
# classes.npy가 배포 환경에 없을 경우 기존 서비스와 동일한 순서를 fallback으로 사용합니다.
DEFAULT_CLASSES = np.array(
    ["angry", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"],
    dtype=object,
)

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
        model_path: str = "emotion_model.pt",
        classes_path: str = "classes.npy",
        device: Optional[torch.device] = None,
        max_length: int = 64,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.classes = (
            np.load(classes_path, allow_pickle=True)
            if os.path.exists(classes_path)
            else DEFAULT_CLASSES.copy()
        )

        self.model = EmotionClassifier(num_classes=len(self.classes))
        state_dict = torch.load(model_path, map_location=self.device)
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
