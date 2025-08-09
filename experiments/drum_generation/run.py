import os
import argparse
import torch
from huggingface_hub import hf_hub_download

# This script assumes the DrUM source code is available in the PYTHONPATH.
# e.g., by cloning the repo: `git clone https://github.com/Burf/DrUM.git drum_src`
# and then running: `export PYTHONPATH=$(pwd)/drum_src:$PYTHONPATH`
try:
    from drum import DrUM
    from diffusers import DiffusionPipeline, StableDiffusion3Pipeline, FluxPipeline
except ImportError:
    print("Error: Could not import DrUM or diffusers.")
    print("Please ensure the DrUM repository is cloned and in your PYTHONPATH, and that diffusers is installed.")
    print("Example: git clone https://github.com/Burf/DrUM.git drum_src && export PYTHONPATH=$(pwd)/drum_src:$PYTHONPATH")
    print("Example: pip install diffusers transformers accelerate safetensors huggingface-hub Pillow tqdm")
    exit(1)

def download_drum_weights(model_dir="./drum_weights"):
    """
    Downloads the necessary DrUM adapter weights from Hugging Face Hub.
    """
    repo_id = "Burf/DrUM"
    files = ["L.safetensors", "H.safetensors", "bigG.safetensors", "T5.safetensors"]
    
    os.makedirs(model_dir, exist_ok=True)
    print(f"Downloading DrUM weights to {model_dir}...")
    
    for filename in files:
        filepath = os.path.join(model_dir, filename)
        
        if os.path.exists(filepath):
            print(f"✅ {filename} - already exists.")
            continue
            
        print(f"📥 {filename} - downloading...")
        try:
            # Download to a temporary location within the dir to handle hf_hub's pathing
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename="weight/" + filename,
                local_dir=model_dir,
                local_dir_use_symlinks=False,
            )
            # Move file to the correct top-level location if necessary
            if downloaded_path != filepath:
                os.rename(downloaded_path, filepath)
                # Clean up empty subdirectories created by the download library
                try:
                    os.rmdir(os.path.join(model_dir, "weight"))
                except OSError: # Fails if not empty, which is fine
                    pass

            print(f"✅ {filename} - success.")
        except Exception as e:
            print(f"❌ {filename} - fail: {e}")
    
    print("🎉 Finished downloading weights!")
    return model_dir

def main():
    parser = argparse.ArgumentParser(
        description="Generate images using DrUM for stylized conditioning as input for VQASynth."
    )
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5", help="Base T2I model ID from Hugging Face.")
    parser.add_argument("--prompt", type=str, default="a living room with a sofa and a coffee table", help="Main prompt for image generation.")
    parser.add_argument("--ref", nargs='+', default=["A minimalist interior design photograph with clean lines and neutral colors."], help="Reference descriptions for style/concept conditioning.")
    parser.add_argument("--weight", nargs='+', type=float, default=[1.0], help="Weight for the reference descriptions.")
    parser.add_argument("--alpha", type=float, default=0.3, help="Alpha scaling factor for DrUM conditioning (range: 0-1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate.")
    parser.add_argument("--save_path", type=str, default="./output", help="Directory to save generated images.")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device ID to use.")

    args = parser.parse_args()

    # --- Environment Setup ---
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    # --- Download DrUM Weights ---
    weights_dir = download_drum_weights()

    # --- Load Pipelines ---
    print(f"Loading base model: {args.model}")
    if "flux" in args.model.lower():
        pipeline = FluxPipeline.from_pretrained(args.model, torch_dtype=dtype)
    elif "stable-diffusion-3" in args.model.lower():
        pipeline = StableDiffusion3Pipeline.from_pretrained(args.model, torch_dtype=dtype)
    else:
        pipeline = DiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype)
    
    pipeline = pipeline.to(device)

    # --- Load DrUM ---
    print("Loading DrUM adapter...")
    drum = DrUM(pipeline, weight=weights_dir)

    # --- Generate Image ---
    print(f'Generating image for prompt: "{args.prompt}" with ref: "{args.ref}"')
    images = drum(
        args.prompt,
        args.ref,
        weight=args.weight,
        alpha=args.alpha,
        seed=args.seed,
        num_images_per_prompt=args.num_images
    )

    # --- Save Results ---
    os.makedirs(args.save_path, exist_ok=True)
    for i, img in enumerate(images):
        safe_prompt = "".join(x for x in args.prompt if x.isalnum() or x in " _-").strip().replace(" ", "_")[:50]
        img.save(os.path.join(args.save_path, f"{safe_prompt}_{i:02d}.png"))

    print(f"Inference complete. {len(images)} image(s) saved to {args.save_path}")

if __name__ == "__main__":
    main()