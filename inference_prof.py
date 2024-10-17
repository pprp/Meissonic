import os
import sys
sys.path.append("./")

import torch
from torchvision import transforms
from src.transformer import Transformer2DModel
from src.pipeline import Pipeline
from src.scheduler import Scheduler
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from diffusers import VQModel

device = 'cuda'

def load_models():
    model_path = "MeissonFlow/Meissonic"
    model = Transformer2DModel.from_pretrained(model_path,subfolder="transformer",)
    vq_model = VQModel.from_pretrained(model_path, subfolder="vqvae", )
    text_encoder = CLIPTextModelWithProjection.from_pretrained(   #using original text enc for stable sampling
                "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
            )
    tokenizer = CLIPTokenizer.from_pretrained(model_path,subfolder="tokenizer",)
    scheduler = Scheduler.from_pretrained(model_path,subfolder="scheduler",)
    pipe = Pipeline(vq_model, tokenizer=tokenizer,text_encoder=text_encoder,transformer=model,scheduler=scheduler)
    return pipe.to(device)

def run_inference(pipe, prompt, negative_prompt, resolution, cfg, steps):
    return pipe(prompt=prompt, negative_prompt=negative_prompt, height=resolution, width=resolution, guidance_scale=cfg, num_inference_steps=steps).images[0]

def main():
    pipe = load_models()

    steps = 64
    CFG = 9
    resolution = 1024 
    negative_prompts = "worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark"

    prompts = [
        "Two actors are posing for a pictur with one wearing a black and white face paint.",
        "A large body of water with a rock in the middle and mountains in the background.",
        "A white and blue coffee mug with a picture of a man on it.",
        "A statue of a man with a crown on his head.",
        "A man in a yellow wet suit is holding a big black dog in the water.",
        "A white table with a vase of flowers and a cup of coffee on top of it.",
        "A woman stands on a dock in the fog.",
        "A woman is standing next to a picture of another woman."
    ]

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/meissonic'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for step, prompt in enumerate(prompts):
            image = run_inference(pipe, prompt, negative_prompts, resolution, CFG, steps)
            image.save(os.path.join(output_dir, f"{prompt[:10]}_{resolution}_{steps}_{CFG}.png"))
            prof.step()

if __name__ == "__main__":
    main()
