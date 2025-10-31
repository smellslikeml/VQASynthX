import argparse
import time
import torch
import lpips
from diffusers import StableDiffusionPipeline
from torchvision.utils import save_image
import os

# Add the vendored sada library to the Python path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from sada import patch


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    # VQASynth-style prompt focusing on spatial relationships
    prompt = "A high-resolution photo of a red cube to the left of a blue sphere on a wooden table."

    # --- Baseline ---
    print("Running baseline inference...")
    baseline_pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16
    ).to("cuda")
    baseline_pipe.enable_xformers_memory_efficient_attention()

    set_random_seed(args.seed)
    start_time = time.time()
    baseline_image = baseline_pipe(
        prompt, num_inference_steps=50, guidance_scale=7.5
    ).images[0]
    baseline_time = time.time() - start_time
    print(f"Baseline generation took: {baseline_time:.2f} seconds")

    baseline_image_tensor = T.ToTensor()(baseline_image).unsqueeze(0)
    save_image(baseline_image_tensor, "baseline_output.png")
    del baseline_pipe
    torch.cuda.empty_cache()

    # --- SADA Accelerated ---
    print("\nRunning SADA-accelerated inference...")
    sada_pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16
    ).to("cuda")
    sada_pipe.enable_xformers_memory_efficient_attention()

    print("Applying SADA patch...")
    patch.apply_patch(
        sada_pipe,
        sx=3,
        sy=3,
        max_downsample=1,
        acc_range=(10, 47),
        lagrange_int=4,
        lagrange_step=24,
        lagrange_term=4,
        max_fix=1024 * 5,
        max_interval=4,
    )

    set_random_seed(args.seed)
    start_time = time.time()
    sada_image = sada_pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    sada_time = time.time() - start_time
    print(f"SADA generation took: {sada_time:.2f} seconds")

    sada_image_tensor = T.ToTensor()(sada_image).unsqueeze(0)
    save_image(sada_image_tensor, "sada_output.png")

    # --- Comparison ---
    speedup = baseline_time / sada_time
    print(f"\nSpeedup: {speedup:.2f}x")

    print("Evaluating LPIPS distance...")
    loss_fn_alex = lpips.LPIPS(net="alex").to("cuda")
    # Preprocess for LPIPS: scale to [-1, 1]
    baseline_lpips = baseline_image_tensor * 2 - 1
    sada_lpips = sada_image_tensor * 2 - 1
    distance = loss_fn_alex(baseline_lpips.to("cuda"), sada_lpips.to("cuda"))
    print(f"LPIPS Score: {distance.item():.4f}")

    save_image([baseline_image_tensor[0], sada_image_tensor[0]], "comparison.png")
    print("\nSaved baseline_output.png, sada_output.png, and comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark SADA for diffusion model acceleration."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        help="Hugging Face model ID",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    import torchvision.transforms as T

    main(args)
