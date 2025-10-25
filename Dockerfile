# =========================================
#  Базовый образ: минимальный Python + CUDA
# =========================================
FROM python:3.10-slim

# Добавляем CUDA библиотеки (для работы GPU)
RUN apt-get update && apt-get install -y wget gnupg && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-12-1 cuda-compat-12-1 \
        libcublas-12-1 libcufft-12-1 libcurand-12-1 libcusparse-12-1 libcusolver-12-1 \
        ffmpeg git wget && \
    rm -rf /var/lib/apt/lists/*

# ===============================
# Установка совместимых библиотек
# ===============================
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        numpy==1.26.4 \
        torch==2.1.2+cu121 \
        torchvision==0.16.2+cu121 \
        xformers==0.0.23.post1 \
        -f https://download.pytorch.org/whl/torch_stable.html

# Проверка версий (без GPU)
RUN python3 -c "import torch, torchvision, xformers; \
print(f'✅ Torch {torch.__version__} | TV {torchvision.__version__} | XF {xformers.__version__}'); \
print('⚙️ GPU активируется только при запуске (--gpus all)')"

# ===============================
# Установка зависимостей приложения
# ===============================
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ===============================
# Копирование приложения
# ===============================
COPY app.py .

# ===============================
# Запуск FastAPI
# ===============================
EXPOSE 8080
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

