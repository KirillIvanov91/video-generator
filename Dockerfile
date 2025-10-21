# Используем CUDA-окружение с поддержкой GPU
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Устанавливаем Python, FFmpeg (для видео) и системные зависимости
RUN apt update && apt install -y \
    python3 python3-pip ffmpeg git wget \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /app

# Копируем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Загружаем модель Zeroscope v2 XL при сборке
RUN python3 - <<'EOF'
from diffusers import TextToVideoSDPipeline
import torch

print("⬇️ Загружаем модель Zeroscope v2 XL (Text-to-Video)...")

pipe = TextToVideoSDPipeline.from_pretrained(
    "cerspense/zeroscope_v2_XL",
    torch_dtype=torch.float16
)

print("✅ Модель закэширована в Hugging Face.")
EOF

# Копируем серверный код
COPY app.py .

# Открываем порт для FastAPI
EXPOSE 8000

# Запускаем FastAPI сервер
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
