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
import time
from quantize_fp8 import quantize_transformer2d_and_dispatch_float8, F8Linear, recursive_swap_linears

device = 'cuda'

def load_fp8_models(use_gpu=True):
    model_path = "MeissonFlow/Meissonic"
    model = Transformer2DModel.from_pretrained(model_path, subfolder="transformer")
    vq_model = VQModel.from_pretrained(model_path, subfolder="vqvae")
    text_encoder = CLIPTextModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    scheduler = Scheduler.from_pretrained(model_path, subfolder="scheduler")

    # Quantize models to FP8
    model_fp8 = quantize_transformer2d_and_dispatch_float8(
        model,
        device=torch.device(device if use_gpu else 'cpu'),
        float8_dtype=torch.float8_e4m3fn,
        input_float8_dtype=torch.float8_e5m2,
        offload_transformer=False,
        swap_linears_with_cublaslinear=True,
        transformer_dtype=torch.float16
    )
    
    # Convert other models to FP8
    vq_model = convert_model_to_fp8(vq_model)
    text_encoder = convert_model_to_fp8(text_encoder)

    pipe = Pipeline(vq_model, tokenizer=tokenizer, text_encoder=text_encoder, transformer=model_fp8, scheduler=scheduler)
    return pipe.to(device if use_gpu else 'cpu')

def convert_model_to_fp8(model):
    recursive_swap_linears(
        model,
        float8_dtype=torch.float8_e4m3fn,
        input_float8_dtype=torch.float8_e5m2
    )
    return model

def run_inference(pipe, prompt, negative_prompt, resolution, cfg, steps):
    return pipe(prompt=prompt, negative_prompt=negative_prompt, height=resolution, width=resolution, guidance_scale=cfg, num_inference_steps=steps).images[0]

def main():
    steps = 64
    CFG = 9
    resolution = 1024 
    negative_prompts = "worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark"

    prompts = [
        "Two actors are posing for a pictur with one wearing a black and white face paint.",
        "A large body of water with a rock in the middle and mountains in the background.",
        "A white and blue coffee mug with a picture of a man on it.",
    ]

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    # FP8 precision inference
    pipe_fp8 = load_fp8_models()
    if pipe_fp8 is not None:
        start_time = time.time()
        for prompt in prompts:
            image = run_inference(pipe_fp8, prompt, negative_prompts, resolution, CFG, steps)
            image.save(os.path.join(output_dir, f"{prompt[:10]}_{resolution}_{steps}_{CFG}_fp8.png"))
        total_time_fp8 = time.time() - start_time
        print(f"Total inference time (FP8): {total_time_fp8:.2f} seconds")

if __name__ == "__main__":
    main()
