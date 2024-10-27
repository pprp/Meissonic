import os
import sys
sys.path.append("./")

import torch
from src.transformer import Transformer2DModel
from src.scheduler import Scheduler
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
import time
from optimum.quanto import quantize, qint4, qint8, freeze

device = 'cuda'


def load_models():
    from src.pipeline import Pipeline
    from diffusers import VQModel

    model_path = "MeissonFlow/Meissonic"
    dtype = torch.bfloat16
    model = Transformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=dtype)
    vq_model = VQModel.from_pretrained(model_path, subfolder="vqvae", torch_dtype=dtype)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        torch_dtype=dtype
    )
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", torch_dtype=dtype)
    scheduler = Scheduler.from_pretrained(model_path, subfolder="scheduler")
    
    pipe = Pipeline(vq_model, tokenizer=tokenizer, text_encoder=text_encoder, transformer=model, scheduler=scheduler)
    
    include_modules = []
    for name, module in pipe.transformer.named_modules():
        if isinstance(module, torch.nn.Linear):
            include_modules.append(name)
    quantize(pipe.transformer, weights=qint4, include=include_modules)
    freeze(pipe.transformer)
    return pipe.to(device)

def run_inference(pipe, prompt, negative_prompt, resolution, cfg, steps):
    return pipe(prompt=prompt, negative_prompt=negative_prompt, height=resolution, width=resolution, guidance_scale=cfg, num_inference_steps=steps).images[0]

if __name__ == "__main__":
    steps = 64
    CFG = 9
    resolution = 1024 
    negative_prompts = "worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark"

    prompts = [
        "Two actors are posing for a pictur with one wearing a black and white face paint.",
        "A large body of water with a rock in the middle and mountains in the background.",
        "A white and blue coffee mug with a picture of a man on it.",
        # "The sun is setting over a city skyline with a river in the foreground.",
        # "A black and white cat with blue eyes.", 
        # "Three boats in the ocean with a rainbow in the sky.", 
        # "A robot playing the piano.",
        # "A cat wearing a hat.",
        # "A dog in a jungle.",
    ]

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    pipe = load_models()
    start_time = time.time()
    total_memory_used = 0
    for i, prompt in enumerate(prompts):
        torch.cuda.reset_peak_memory_stats()
        image_start_time = time.time()
        image = run_inference(pipe, prompt, negative_prompts, resolution, CFG, steps)
        image_end_time = time.time()
        image.save(os.path.join(output_dir, f"{prompt[:10]}_{resolution}_{steps}_{CFG}.png"))
        
        memory_used = torch.cuda.max_memory_reserved() / (1024 ** 3)  # Convert to GB
        total_memory_used += memory_used
        
        print(f"Image {i+1} time: {image_end_time - image_start_time:.2f} seconds")
        print(f"Image {i+1} max memory used: {memory_used:.2f} GB")
    
    total_time = time.time() - start_time
    avg_memory_used = total_memory_used / len(prompts)
    print(f"Total inference time: {total_time:.2f} seconds")
    print(f"Average memory used per image: {avg_memory_used:.2f} GB")
