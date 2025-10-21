from math import e
from fastapi import FastAPI
from pydantic import BaseModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
import torch, uuid, os, imageio
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth import TextToVideoSDPipeline
from torchvision.io import write_video
from pathlib import Path


app = FastAPI()

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)


print("🔄 Загружается модель...")
pipe = TextToVideoSDPipeline.from_pretrained(
    "cerspense/zeroscope_v2_XL",
    torch_dtype=torch.float16
    
).to("cuda")
print("✅ Модель готова к работе!")

# Оптимизация памяти и скорости
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()
pipe.to("cuda")

class VideoRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate(req: VideoRequest):
    print(f"GPU доступен: {torch.cuda.is_available()}")
    print(f"Текущее устройство: {next(pipe.parameters()).device}")
    print(f"🎬 Генерация: {req.prompt}")
    video_tensor = pipe(req.prompt, num_inference_steps=50).videos[0]
    #video_tensor = pipe(req.prompt, num_frames=60).videos[0]  # Получаем тензор (T, H, W, C)
    filename = f"{uuid.uuid4().hex}.mp4"
    output_path = output_dir / filename
    imageio.mimsave(output_path, video_tensor, fps=8)
    print(f"✅ Видео сохранено: {output_path}")
    #os.makedirs("output", exist_ok=True)
    #path = f"output/{filename}"
    #write_video(path, video_tensor.permute(0, 3, 1, 2), fps=8)  # Перемещаем каналы и задаем FPS
    return {"video_path": str(output_path)}

@app.get("/")
async def read_root():
    return {"message": "Сервис генерации видео запущен! Используйте эндпоинт /generate."}





#docker run --gpus all -p 8000:8000 -v ${PWD}:/app zeroscope-server
