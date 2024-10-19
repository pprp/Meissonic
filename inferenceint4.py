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

from torchao.quantization.quant_api import (
    quantize_,
    int4_weight_only, #A16W4 Weight-only 
    int8_weight_only, #A16W8 Weight-only 
    int8_dynamic_activation_int8_semi_sparse_weight,  
    int8_dynamic_activation_int8_weight,  # W8A8 INT8
    float8_weight_only, # A8W8 FP8
    float8_dynamic_activation_float8_weight,  # W8A8 FP8 Dynamic Quantization;
    fpx_weight_only, # A16W6 fp Weight-only 
)
import torchao 

device = 'cuda'

def get_quantization_method(method, group_size=32):
    quantization_methods = {
        'int4': lambda: int4_weight_only(group_size=group_size),
        'int4_hqq': lambda: int4_weight_only(group_size=group_size, use_hqq=True),
        'int8': int8_weight_only,
        'int8_dynamic': int8_dynamic_activation_int8_weight,
        'int8_semi_sparse': int8_dynamic_activation_int8_semi_sparse_weight,
        'float8': float8_weight_only,
        'float8_dynamic': float8_dynamic_activation_float8_weight,
        'a16w6': lambda: fpx_weight_only(3, 2),
    }
    return quantization_methods.get(method, None)

def load_models(precision, quantization_method=None, group_size=32):
    model_path = "MeissonFlow/Meissonic"
    dtype = torch.bfloat16
    model = Transformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=dtype)
    vq_model = VQModel.from_pretrained(model_path, subfolder="vqvae", torch_dtype=dtype)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        torch_dtype=dtype
    )
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    scheduler = Scheduler.from_pretrained(model_path, subfolder="scheduler")
    
    if quantization_method:
        quant_method = get_quantization_method(quantization_method, group_size)
        if quant_method:
            quantize_(model, quant_method())
        else:
            print(f"Unsupported quantization method: {quantization_method}")

    if precision == 'fp8':
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
    return pipe.to(device)

def run_inference(pipe, prompt, negative_prompt, resolution, cfg, steps):
    return pipe(prompt=prompt, negative_prompt=negative_prompt, height=resolution, width=resolution, guidance_scale=cfg, num_inference_steps=steps).images[0]

def main(precision, quantization_method, group_size):
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

    pipe = load_models(precision, quantization_method, group_size)
    start_time = time.time()
    total_memory_used = 0
    for i, prompt in enumerate(prompts):
        torch.cuda.reset_peak_memory_stats()
        image_start_time = time.time()
        image = run_inference(pipe, prompt, negative_prompts, resolution, CFG, steps)
        image_end_time = time.time()
        image.save(os.path.join(output_dir, f"{prompt[:10]}_{resolution}_{steps}_{CFG}_{quantization_method}.png"))
        
        memory_used = torch.cuda.max_memory_reserved() / (1024 ** 3)  # Convert to GB
        total_memory_used += memory_used
        
        print(f"Image {i+1} time: {image_end_time - image_start_time:.2f} seconds")
        print(f"Image {i+1} max memory used: {memory_used:.2f} GB")
    
    total_time = time.time() - start_time
    avg_memory_used = total_memory_used / len(prompts)
    print(f"Total inference time ({precision}, {quantization_method}): {total_time:.2f} seconds")
    print(f"Average memory used per image: {avg_memory_used:.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with specified precision and quantization method.")
    parser.add_argument("--precision", type=str, choices=['fp32', 'fp16', 'bf16', 'fp8'], default='fp32',
                        help="Precision to use for inference (fp32, fp16, bf16, or fp8)")
    parser.add_argument("--quantization", type=str, choices=['int4', 'int4_hqq', 'int8', 'int8_dynamic', 'int8_semi_sparse', 'float8', 'float8_dynamic', 'a16w6'], 
                        help="Quantization method to use")
    parser.add_argument("--group_size", type=int, default=32,
                        help="Group size for int4 quantization")
    args = parser.parse_args()
    main(args.precision, args.quantization, args.group_size)
