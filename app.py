
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
import torch, uuid
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth import TextToVideoSDPipeline
from torchvision.io import write_video
from pathlib import Path

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {device_name}, VRAM: {vram_gb:.2f} GB")
else:
    print("⚠️ GPU не обнаружен — будет использован CPU")

app = FastAPI()

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)


print("🔄 Загружается модель...")

try:
    pipe = TextToVideoSDPipeline.from_pretrained(
    "cerspense/zeroscope_v2_XL",
        torch_dtype=torch.float16
    ).to("cuda")

    print("✅ Модель готова к работе!")

    # Оптимизация памяти и скорости
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    
except Exception as exc:
    raise RuntimeError(f"Ошибка загрузки модели: {exc}")



class VideoRequest(BaseModel):
    prompt: str




@app.post("/generate")
async def generate(req: VideoRequest):
    try:
        
        print(f"🎬 Генерация: {req.prompt}")
            
        # Генерация видео (T, H, W, C)
        video_tensor = pipe(req.prompt, num_inference_steps=30).videos[0]

        # Перестановка осей (T, C, H, W) для torchvision
        video_tensor = video_tensor.permute(0, 3, 1, 2)

        filename = f"{uuid.uuid4().hex}.mp4"
        output_path = output_dir / filename

        # Сохранение видео с указанием FPS
        write_video(str(output_path), video_tensor, fps=8)
        print(f"✅ Видео сохранено: {output_path}")
        
        return {"video_path": str(output_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Запрос поступил, но генерация не началась: {str(e)}")
    
    
@app.get("/")
async def read_root():
    return {"message": "Сервис генерации видео запущен! Используйте эндпоинт /generate."}





#docker run --gpus all -p 8000:8000 -v ${PWD}:/app zeroscope-server
