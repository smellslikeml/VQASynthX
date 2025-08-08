import time
import torch
import lpips
from diffusers import FluxPipeline
from torchvision.utils import save_image
import torchvision.transforms as T

# Import the SADA patch module
from sada import patch

def set_random_seed(seed):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    """
    Main function to run the evaluation generation experiment.
    Generates an image with a baseline diffusion pipeline and an SADA-accelerated one,
    then compares their speed and visual similarity.
    """
    # --- Configuration ---
    MODEL_ID = "black-forest-labs/FLUX.1-dev"
    PROMPT = "A high-resolution photo of a red cube directly on top of a large blue sphere. They are on a plain white surface in a brightly lit studio."
    SEED = 42
    HEIGHT = 1024
    WIDTH = 1024
    NUM_INFERENCE_STEPS = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16

    set_random_seed(SEED)
    print("--- Ravine Evaluator: SADA Acceleration Experiment ---")
    print(f"Using model: {MODEL_ID}")
    print(f"Using device: {DEVICE}")

    # --- Baseline Generation ---
    print("\n1. Running baseline generation...")
    baseline_pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    
    # Warmup
    _ = baseline_pipe(PROMPT, height=HEIGHT, width=WIDTH, num_inference_steps=2, output_type='pt').images
    
    start_time = time.time()
    set_random_seed(SEED)
    baseline_output = baseline_pipe(
        PROMPT,
        height=HEIGHT,
        width=WIDTH,
        guidance_scale=3.5,
        num_inference_steps=NUM_INFERENCE_STEPS,
        output_type='pt'
    ).images
    baseline_time = time.time() - start_time
    print(f"Baseline generation took: {baseline_time:.2f} seconds")

    del baseline_pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- SADA-accelerated Generation ---
    print("\n2. Running SADA-accelerated generation...")
    sada_pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    
    # Apply the SADA patch
    patch.apply_patch(
        sada_pipe,
        max_downsample=0,
        acc_range=(10, 47),
        latent_size=(HEIGHT // 16, WIDTH // 16),
        lagrange_int=4,
        lagrange_step=20,
        lagrange_term=3,
        max_fix=0,
        max_interval=4
    )

    # Warmup
    _ = sada_pipe(PROMPT, height=HEIGHT, width=WIDTH, num_inference_steps=2, output_type='pt').images
    patch.reset_cache(sada_pipe)

    start_time = time.time()
    set_random_seed(SEED)
    sada_output = sada_pipe(
        PROMPT,
        height=HEIGHT,
        width=WIDTH,
        guidance_scale=3.5,
        num_inference_steps=NUM_INFERENCE_STEPS,
        output_type='pt'
    ).images
    sada_time = time.time() - start_time
    print(f"SADA generation took: {sada_time:.2f} seconds")

    # --- Analysis ---
    print("\n3. Analyzing results...")
    speedup = baseline_time / sada_time
    print(f"Speedup: {speedup:.2f}x")

    save_image([baseline_output[0], sada_output[0]], "ravine_comparison.png")
    print("Saved comparison image to 'ravine_comparison.png'")

    # Calculate LPIPS
    loss_fn_alex = lpips.LPIPS(net='alex').to(DEVICE)
    # Normalize images for LPIPS
    transform = T.Compose([T.Normalize((0.5,), (0.5,))])
    baseline_tensor = transform(baseline_output[0]).to(DEVICE)
    sada_tensor = transform(sada_output[0]).to(DEVICE)
    
    lpips_distance = loss_fn_alex(baseline_tensor.unsqueeze(0), sada_tensor.unsqueeze(0))
    print(f"LPIPS distance (lower is better): {lpips_distance.item():.4f}")

if __name__ == "__main__":
    main()
