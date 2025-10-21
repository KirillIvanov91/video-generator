from math import e
from fastapi import FastAPI
from pydantic import BaseModel
from diffusers
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
import torch, uuid, os
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth import TextToVideoSDPipeline
from torchvision.io import write_video

app = FastAPI()

print("🔄 Загружается модель...")
pipe = TextToVideoSDPipeline.from_pretrained(
    "cerspense/zeroscope_v2_XL",
    torch_dtype=torch.float16
    
).pipe.to("cuda")
print("✅ Модель готова!")

class VideoRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate(req: VideoRequest):
    print(f"GPU доступен: {torch.cuda.is_available()}")
    print(f"Текущее устройство: {next(pipe.parameters()).device}")
    filename = f"{uuid.uuid4().hex}.mp4"
    os.makedirs("output", exist_ok=True)
    print(f"🎬 Генерация: {req.prompt}")
    video_tensor = pipe(req.prompt, num_frames=60).videos[0]  # Получаем тензор (T, H, W, C)
    result = pipe(req.prompt, num_frames=60).videos[0]
    path = f"output/{filename}"
    write_video(path, video_tensor.permute(0, 3, 1, 2), fps=8)  # Перемещаем каналы и задаем FPS
    result.save(path)
    return {"video_path": path}

@app.get("/")
async def read_root():
    return {"message": "Сервис генерации видео запущен! Используйте эндпоинт /generate."}





#docker run --gpus all -p 8000:8000 -v ${PWD}:/app zeroscope-server
