import argparse
import logging
import time
import torch
import lpips

from torchvision.utils import save_image
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
import torchvision.transforms as T

# SADA is installed via requirements.txt
from sada import patch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    # Setup
    model_id = "stabilityai/stable-diffusion-2-1-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_random_seed(args.seed)

    # --- Baseline Run ---
    logging.info("Loading baseline pipeline...")
    base_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16")
    base_pipe.scheduler = EulerDiscreteScheduler.from_config(base_pipe.scheduler.config)
    base_pipe = base_pipe.to(device)

    logging.info("Warming up GPU for baseline...")
    _ = base_pipe(args.prompt, num_inference_steps=50, guidance_scale=7.5, output_type='pt').images

    logging.info("Running baseline inference...")
    start_time = time.time()
    set_random_seed(args.seed)
    baseline_output = base_pipe(args.prompt, num_inference_steps=50, guidance_scale=7.5, output_type='pt').images
    baseline_time = time.time() - start_time
    logging.info(f"Baseline generation took: {baseline_time:.2f} seconds")

    del base_pipe
    torch.cuda.empty_cache()

    # --- SADA Run ---
    logging.info("Loading pipeline for SADA...")
    sada_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16")
    sada_pipe.scheduler = EulerDiscreteScheduler.from_config(sada_pipe.scheduler.config)
    sada_pipe = sada_pipe.to(device)

    logging.info("Applying SADA patch...")
    patch.apply_patch(sada_pipe,
                      sx=3, sy=3,
                      max_downsample=1,
                      acc_range=(10, 47),
                      lagrange_int=4,
                      lagrange_step=24,
                      lagrange_term=4,
                      max_fix=1024 * 5,
                      max_interval=4
                    )

    logging.info("Warming up GPU for SADA...")
    set_random_seed(args.seed)
    _ = sada_pipe(args.prompt, num_inference_steps=50, guidance_scale=7.5, output_type='pt').images
    patch.reset_cache(sada_pipe)

    logging.info("Running SADA inference...")
    start_time = time.time()
    set_random_seed(args.seed)
    sada_output = sada_pipe(args.prompt, num_inference_steps=50, guidance_scale=7.5, output_type='pt').images
    sada_time = time.time() - start_time
    logging.info(f"SADA generation took: {sada_time:.2f} seconds")

    # --- Comparison ---
    speedup = baseline_time / sada_time if sada_time > 0 else float('inf')
    logging.info(f"Speedup: {speedup:.2f}x")

    save_image([baseline_output[0], sada_output[0]], "sada_vs_baseline.png")
    logging.info("Saved comparison image to sada_vs_baseline.png")

    logging.info("Evaluating LPIPS distance...")
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    # Normalize images for LPIPS
    transform = T.Compose([T.Normalize((0.5,), (0.5,))])
    baseline_tensor = transform(baseline_output).to(device)
    sada_tensor = transform(sada_output).to(device)
    distance = loss_fn_alex(baseline_tensor, sada_tensor)
    logging.info(f"LPIPS distance: {distance.item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark SADA against baseline Stable Diffusion.")
    parser.add_argument("--prompt", type=str, default="A high-quality photo of an astronaut riding a horse.", help="Prompt for image generation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    main(args)
