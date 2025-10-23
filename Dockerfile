
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04


RUN apt update && apt install -y \
    python3 python3-pip ffmpeg git wget libcudnn8=8.9.2.26-1+cuda12.1 \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y ffmpeg
RUN pip install --no-cache-dir torch==2.2.1+cu121 torchvision==0.17.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir xformers==0.0.23.post1 -f https://download.pytorch.org/whl/torch_stable.html
WORKDIR /app



COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


#RUN python3 - <<'EOF'
#from diffusers import TextToVideoSDPipeline
#import torch

#print("⬇️ Загружаем модель Zeroscope v2 XL (Text-to-Video)...")

#pipe = TextToVideoSDPipeline.from_pretrained(
#    "cerspense/zeroscope_v2_XL",
#    torch_dtype=torch.float16
#)

#print("✅ Модель закэширована в Hugging Face.")
#EOF

COPY app.py .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
