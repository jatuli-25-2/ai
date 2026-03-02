FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install only minimal OS deps used by common Python wheels/runtime.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Force CPU-only torch to avoid massive CUDA image bloat.
RUN set -eux; \
    grep -vE '^torch([<>=!~ ]|$)' requirements.txt > /tmp/requirements.no-torch.txt; \
    pip install --upgrade pip; \
    pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1.0"; \
    pip install -r /tmp/requirements.no-torch.txt

COPY src ./src
COPY data ./data
COPY emotion_model.pt ./emotion_model.pt

EXPOSE 8000

CMD ["uvicorn", "--app-dir", "src", "ai_main_code_server:app", "--host", "0.0.0.0", "--port", "8000"]
