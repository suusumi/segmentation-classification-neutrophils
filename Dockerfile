FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV YOLO_CONFIG_DIR=/tmp

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.runtime.txt ./
COPY src ./src
COPY configs ./configs
COPY models ./models

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.12.0+cpu \
        torchvision==0.27.0+cpu \
    && pip install --no-cache-dir -r requirements.runtime.txt

RUN mkdir -p data/raw data/interim data/processed outputs

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
