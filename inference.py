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

def load_fp16_models():
    model_path = "MeissonFlow/Meissonic"
    model = Transformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=torch.float16)
    vq_model = VQModel.from_pretrained(model_path, subfolder="vqvae", torch_dtype=torch.float16)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        torch_dtype=torch.float16
    )
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    scheduler = Scheduler.from_pretrained(model_path, subfolder="scheduler")
    pipe = Pipeline(vq_model, tokenizer=tokenizer, text_encoder=text_encoder, transformer=model, scheduler=scheduler)
    return pipe.to(device)

def load_int4_models(use_gpu=True):
    try:
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig

        model_path = "MeissonFlow/Meissonic"
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model = Transformer2DModel.from_pretrained(model_path, subfolder="transformer", quantization_config=quantization_config)
        vq_model = VQModel.from_pretrained(model_path, subfolder="vqvae", quantization_config=quantization_config)
        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            quantization_config=quantization_config
        )
        tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        scheduler = Scheduler.from_pretrained(model_path, subfolder="scheduler")
        pipe = Pipeline(vq_model, tokenizer=tokenizer, text_encoder=text_encoder, transformer=model, scheduler=scheduler)
        return pipe.to(device if use_gpu else 'cpu')
    except ImportError:
        print("bitsandbytes package not found. INT4 quantization is not available.")
        return None
    except ValueError as e:
        if "sequential model offloading" in str(e):
            print("Sequential CPU offloading detected. Forcing CPU usage for INT4 models.")
            return pipe.to('cpu')
        else:
            raise e


def run_inference(pipe, prompt, negative_prompt, resolution, cfg, steps):
    return pipe(prompt=prompt, negative_prompt=negative_prompt, height=resolution, width=resolution, guidance_scale=cfg, num_inference_steps=steps).images[0]

def main():
    # Regular precision
    pipe = load_models()
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

    # Regular precision inference
    # start_time = time.time()
    # for prompt in prompts:
    #     image = run_inference(pipe, prompt, negative_prompts, resolution, CFG, steps)
    #     image.save(os.path.join(output_dir, f"{prompt[:10]}_{resolution}_{steps}_{CFG}.png"))
    # total_time = time.time() - start_time
    # print(f"Total inference time (regular precision): {total_time:.2f} seconds")

    # FP16 precision inference
    # pipe_fp16 = load_fp16_models()
    # start_time = time.time()
    # for prompt in prompts:
    #     image = run_inference(pipe_fp16, prompt, negative_prompts, resolution, CFG, steps)
    #     image.save(os.path.join(output_dir, f"{prompt[:10]}_{resolution}_{steps}_{CFG}_fp16.png"))
    # total_time_fp16 = time.time() - start_time
    # print(f"Total inference time (FP16): {total_time_fp16:.2f} seconds")

    # INT4 precision inference
    pipe_int4 = load_int4_models()
    if pipe_int4 is not None:
        start_time = time.time()
        for prompt in prompts:
            image = run_inference(pipe_int4, prompt, negative_prompts, resolution, CFG, steps)
            image.save(os.path.join(output_dir, f"{prompt[:10]}_{resolution}_{steps}_{CFG}_int4.png"))
        total_time_int4 = time.time() - start_time
        print(f"Total inference time (INT4): {total_time_int4:.2f} seconds")

if __name__ == "__main__":
    main()
