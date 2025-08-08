import time
import torch
import logging
from torchvision.utils import save_image
from diffusers import FluxPipeline

# SADA is the library from the source repository.
# It can be installed via: pip install git+https://github.com/Ting-Justin-Jiang/sada-icml.git
from sada import patch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_random_seed(seed):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    """
    This script demonstrates the SADA acceleration technique on the FLUX.1-dev model.
    It runs a baseline generation, then applies the SADA patch and runs an accelerated
    generation to compare speed and output quality.
    """
    # --- Configuration ---
    MODEL_ID = "black-forest-labs/FLUX.1-dev"
    PROMPT = "A cinematic photo of a robot orchestra playing in a futuristic, neon-lit city."
    SEED = 42
    NUM_INFERENCE_STEPS = 50
    HEIGHT = 1024
    WIDTH = 1024
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Baseline Generation ---
    logging.info("Loading pipeline for baseline generation...")
    baseline_pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(DEVICE)
    
    logging.info("Warming up GPU for baseline...")
    set_random_seed(SEED)
    _ = baseline_pipe(PROMPT, num_inference_steps=2, output_type='pt').images

    logging.info("Running baseline generation...")
    set_random_seed(SEED)
    start_time = time.time()
    baseline_output = baseline_pipe(
        PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        height=HEIGHT,
        width=WIDTH,
        output_type='pt'
    ).images
    baseline_time = time.time() - start_time
    logging.info(f"Baseline generation took: {baseline_time:.2f} seconds.")
    
    del baseline_pipe
    torch.cuda.empty_cache()

    # --- SADA Accelerated Generation ---
    logging.info("Loading pipeline for SADA-accelerated generation...")
    sada_pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(DEVICE)

    logging.info("Applying SADA patch...")
    # These parameters are adapted from the source repository's flux_demo.py
    patch.apply_patch(sada_pipe,
                      max_downsample=0,
                      acc_range=(10, 47),
                      latent_size=(HEIGHT // 16, WIDTH // 16),
                      lagrange_int=4,
                      lagrange_step=20,
                      lagrange_term=3,
                      max_fix=0,
                      max_interval=4)

    logging.info("Warming up GPU for SADA run...")
    set_random_seed(SEED)
    _ = sada_pipe(PROMPT, num_inference_steps=2, output_type='pt').images
    patch.reset_cache(sada_pipe) # Reset SADA cache after warmup

    logging.info("Running SADA-accelerated generation...")
    set_random_seed(SEED)
    start_time = time.time()
    sada_output = sada_pipe(
        PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        height=HEIGHT,
        width=WIDTH,
        output_type='pt'
    ).images
    sada_time = time.time() - start_time
    logging.info(f"SADA-accelerated generation took: {sada_time:.2f} seconds.")

    # --- Comparison and Output ---
    speedup = baseline_time / sada_time if sada_time > 0 else float('inf')
    logging.info(f"\n--- Results ---")
    logging.info(f"Baseline Time: {baseline_time:.2f}s")
    logging.info(f"SADA Time:     {sada_time:.2f}s")
    logging.info(f"Speedup:       {speedup:.2f}x")

    output_filename = "sada_comparison_output.png"
    save_image([baseline_output[0], sada_output[0]], output_filename)
    logging.info(f"Saved baseline (left) and SADA (right) images to {output_filename}")

if __name__ == "__main__":
    main()
