from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import DiffusionPipeline
import torch, uuid, os

app = FastAPI()

print("🔄 Загружается модель...")
pipe = DiffusionPipeline.from_pretrained(
    "cerspense/zeroscope_v2_XL",
    torch_dtype=torch.float16,
    
).to("cuda")
print("✅ Модель готова!")

class VideoRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate(req: VideoRequest):
    filename = f"{uuid.uuid4().hex}.mp4"
    os.makedirs("output", exist_ok=True)
    print(f"🎬 Генерация: {req.prompt}")
    result = pipe(req.prompt, num_frames=120).videos[0]
    path = f"output/{filename}"
    result.save(path)
    return {"video_path": path}
