FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY alembic.ini ./
COPY alembic ./alembic
COPY scripts ./scripts

RUN pip install --no-cache-dir -e ".[postgres]"

ENV PYTHONUNBUFFERED=1
ENV WEIGHTS_DIR=/app/weights
ENV DATA_DIR=/app/data
ENV RUNS_DIR=/app/runs

RUN python scripts/download_weights.py --models yolov8n yolov5s

EXPOSE 8000

CMD ["disaster-vision-api"]
