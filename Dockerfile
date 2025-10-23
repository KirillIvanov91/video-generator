
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Установка Python, pip, ffmpeg, git и libcudnn
RUN apt update && apt install -y \
    python3 python3-pip ffmpeg git wget libcudnn8=8.9.2.26-1+cuda12.1 \
    && rm -rf /var/lib/apt/lists/*

# Установка PyTorch + torchvision с CUDA 12.1
RUN pip install --no-cache-dir torch==2.2.1+cu121 torchvision==0.17.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html

# Сборка xFormers из исходников под установленный PyTorch
RUN git clone https://github.com/facebookresearch/xformers.git \
    && cd xformers \
    && pip install -e . \
    && cd .. \
    && rm -rf xformers

# Создание рабочей директории
WORKDIR /app

# Копирование requirements.txt и установка остальных зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование приложения
COPY app.py .

# Открытие порта
EXPOSE 8080

# Команда запуска
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]