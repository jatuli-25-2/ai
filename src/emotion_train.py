# emotion_train.py
# 학과 서버용 KoBERT 감정분석 학습 코드

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from kobert_transformers import get_kobert_model, get_tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 데이터 로드 및 전처리
data_path = "./data/emotion_data.csv"  # 학과 서버 경로에 맞게 조정
df = pd.read_csv(data_path, encoding="cp949")[["발화문", "1번 감정"]].dropna()

# 문자열 → 숫자 라벨 인코딩
label_encoder = LabelEncoder()
df["1번 감정"] = label_encoder.fit_transform(df["1번 감정"])
texts = df["발화문"].tolist()
labels = df["1번 감정"].tolist()

# 학습 / 검증 분할
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.1, random_state=42, stratify=labels
)

# Dataset 정의
class KoBERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            add_special_tokens=True,
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

# 모델 정의
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

# 학습 준비
tokenizer = get_tokenizer()
train_dataset = KoBERTDataset(train_texts, train_labels, tokenizer)
val_dataset = KoBERTDataset(val_texts, val_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

model = EmotionClassifier(num_classes=len(label_encoder.classes_)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# 학습 루프
best_val_loss = float("inf")
early_stop_count = 0
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # 검증 단계
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = correct / len(val_dataset)

    print(f"Epoch {epoch+1} Summary:")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "kobert_emotion_best.pt")
        np.save("classes.npy", label_encoder.classes_)
        print("Validation loss improved → Model saved!")
        early_stop_count = 0
    else:
        early_stop_count += 1
        print(f"Validation loss did not improve ({early_stop_count}/2)")
        if early_stop_count >= 2:
            print("Early stopping triggered. Training stopped.")
            break

# GPU 메모리 정리
torch.cuda.empty_cache()
del model
print("Training completed!")
