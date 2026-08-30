# 모델 파일 안내

이 디렉터리에는 자투리 AI 서버에서 사용하는 KoBERT 감정분류 모델 산출물을 배치합니다.

## 필요한 파일

```text
models/
├── emotion_model.pt
└── classes.npy
```

- `emotion_model.pt`
  - `src/emotion_train.py`로 Fine-tuning한 7개 감정 분류 KoBERT 모델의 가중치 파일입니다.
- `classes.npy`
  - 학습 시 `LabelEncoder.classes_`를 저장한 파일로, 모델 출력 인덱스와 실제 감정 라벨의 순서를 일치시키는 데 사용합니다.

## 생성 방법

프로젝트 루트에서 감정분류 학습 코드를 실행하면 검증 손실이 개선될 때 위 두 파일이 이 디렉터리에 저장됩니다.

```bash
python src/emotion_train.py
```

## 실행 시 주의사항

모델 파일은 용량이 크고 학습 결과물에 해당하므로 Git 저장소에는 포함하지 않습니다. AI 서버를 실행하기 전에 `emotion_model.pt`와 `classes.npy`를 `models/` 디렉터리에 배치해야 합니다.

기본 경로는 다음과 같습니다.

```text
models/emotion_model.pt
models/classes.npy
```

필요한 경우 다음 환경변수로 경로를 변경할 수 있습니다.

```env
EMOTION_MODEL_PATH=/path/to/emotion_model.pt
EMOTION_CLASSES_PATH=/path/to/classes.npy
```

Docker로 실행할 때는 모델 파일을 이미지에 포함하지 않고 `models/` 디렉터리를 볼륨으로 연결하는 방식을 권장합니다.
