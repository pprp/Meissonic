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
import argparse
from quantize_fp8 import quantize_transformer2d_and_dispatch_float8, recursive_swap_linears

# torch.compile failed with error:
# torch._inductor.config.conv_1x1_as_mm = True
# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.epilogue_fusion = False
# torch._inductor.config.coordinate_descent_check_all_directions = True

device = 'cuda'

def load_models(precision):
    model_path = "MeissonFlow/Meissonic"
    if precision == 'fp32':
        dtype = torch.float32
    elif precision == 'fp16':
        dtype = torch.float16
    elif precision == 'bf16':
        dtype = torch.bfloat16
    elif precision == 'fp8':
        dtype = torch.float16  # We'll use float16 as a base for fp8 quantization
    else:
        raise ValueError("Invalid precision. Choose 'fp32', 'fp16', 'bf16', or 'fp8'.")

    model = Transformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=dtype)
    vq_model = VQModel.from_pretrained(model_path, subfolder="vqvae", torch_dtype=dtype)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        torch_dtype=dtype
    )
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    scheduler = Scheduler.from_pretrained(model_path, subfolder="scheduler")

    if precision == 'fp8':
        # Quantize vq vae model
        recursive_swap_linears(vq_model)

        # Quantize transformer model to FP8
        model = quantize_transformer2d_and_dispatch_float8(
            model,
            device=torch.device(device),
            float8_dtype=torch.float8_e4m3fn,
            input_float8_dtype=torch.float8_e5m2,
            offload_transformer=False,
            swap_linears_with_cublaslinear=True,
            transformer_dtype=torch.float16
        )
    
    pipe = Pipeline(vq_model, tokenizer=tokenizer, text_encoder=text_encoder, transformer=model, scheduler=scheduler)
    # fuse qkv projections
    # pipe.transformer.fuse_qkv_projections() Error
    
    # change the memory layout to adapt to the torch.compile 
    # pipe.transformer.to(memory_format=torch.channels_last)
    # pipe.vqvae.to(memory_format=torch.channels_last)
    
    # pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
    # pipe.vqvae.decode = torch.compile(pipe.vqvae.decode, mode="max-autotune", fullgraph=True)

    return pipe.to(device)

def run_inference(pipe, prompt, negative_prompt, resolution, cfg, steps):
    return pipe(prompt=prompt, negative_prompt=negative_prompt, height=resolution, width=resolution, guidance_scale=cfg, num_inference_steps=steps).images[0]

def main(precision):
    steps = 64
    CFG = 9
    resolution = 1024 
    negative_prompts = "worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark"

    prompts = [
        "Two actors are posing for a pictur with one wearing a black and white face paint.",
        "A large body of water with a rock in the middle and mountains in the background.",
        "A white and blue coffee mug with a picture of a man on it.",
        "The sun is setting over a city skyline with a river in the foreground.",
        "A black and white cat with blue eyes.", 
        "Three boats in the ocean with a rainbow in the sky.", 
        "A robot playing the piano.",
        "A cat wearing a hat.",
        "A dog in a jungle.",
    ]

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    pipe = load_models(precision)
    start_time = time.time()
    total_memory_used = 0
    for i, prompt in enumerate(prompts):
        torch.cuda.reset_peak_memory_stats()
        image_start_time = time.time()
        image = run_inference(pipe, prompt, negative_prompts, resolution, CFG, steps)
        image_end_time = time.time()
        image.save(os.path.join(output_dir, f"{prompt[:10]}_{resolution}_{steps}_{CFG}_{precision}.png"))
        
        memory_used = torch.cuda.max_memory_reserved() / (1024 ** 3)  # Convert to GB
        total_memory_used += memory_used
        
        print(f"Image {i+1} time: {image_end_time - image_start_time:.2f} seconds")
        print(f"Image {i+1} max memory used: {memory_used:.2f} GB")
    
    total_time = time.time() - start_time
    avg_memory_used = total_memory_used / len(prompts)
    print(f"Total inference time ({precision}): {total_time:.2f} seconds")
    print(f"Average memory used per image: {avg_memory_used:.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with specified precision.")
    parser.add_argument("--precision", type=str, choices=['fp32', 'fp16', 'bf16', 'fp8'], default='fp32',
                        help="Precision to use for inference (fp32, fp16, bf16, or fp8)")
    args = parser.parse_args()
    main(args.precision)
