import torch
import open_clip
from PIL import Image
import os
import argparse
from tqdm import tqdm

# In-distribution class descriptions relevant to VQASynth's goal of spatial reasoning.
# These prompts help CLIP identify suitable scenes for generating VQA samples.
ID_CLASSES = [
    "a photo of a room with furniture",
    "a photo of a living room",
    "a photo of a kitchen",
    "a photo of a bedroom",
    "a photo of an office",
    "a photo of a warehouse",
    "a photo of a garage",
    "a photo of a street scene with cars and buildings",
    "a photo of a city park with benches",
    "a photo of a furnished indoor space",
    "a photo of an outdoor scene with multiple objects",
    "an image containing several distinct objects on a surface",
]


def setup_model(device="cuda"):
    """
    Loads the CLIP model, preprocessor, and tokenizer.
    Inspired by get_clip_model from COOkeD/model_utils.py
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.to(device)
    model.eval()
    return model, preprocess, tokenizer


def calculate_msp(model, preprocess, tokenizer, image_path, device="cuda"):
    """
    Calculates the Maximum Softmax Probability (MSP) for an image against the defined ID classes.
    This logic is directly inspired by the OOD scoring method in COOkeD/demo.py.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        text_prompts = tokenizer(ID_CLASSES).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_prompts)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Calculate softmax probabilities
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            # Get the maximum probability, which is the MSP score
            msp_score = torch.max(text_probs, dim=1).values.item()

        return msp_score
    except Exception as e:
        print(f"Warning: Could not process image {image_path}. Error: {e}")
        return 0.0  # Treat as OOD if processing fails


def main():
    parser = argparse.ArgumentParser(
        description="Filter a directory of images based on CLIP-based OOD detection."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing images to filter.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save in-distribution images.",
    )
    parser.add_argument(
        "--msp_threshold",
        type=float,
        default=0.20,
        help="Minimum MSP score to be considered in-distribution.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu).",
    )

    args = parser.parse_args()

    print(f"Using device: {args.device}")
    print("Setting up CLIP model...")
    model, preprocess, tokenizer = setup_model(device=args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    image_files = [
        f
        for f in os.listdir(args.input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]

    id_count = 0
    ood_count = 0

    print(f"Processing {len(image_files)} images from {args.input_dir}...")
    for filename in tqdm(image_files):
        input_path = os.path.join(args.input_dir, filename)

        msp_score = calculate_msp(
            model, preprocess, tokenizer, input_path, device=args.device
        )

        if msp_score >= args.msp_threshold:
            id_count += 1
            output_path = os.path.join(args.output_dir, filename)
            # Use symlink to avoid duplicating data. Use shutil.copy for actual copy.
            if not os.path.exists(output_path):
                os.symlink(os.path.abspath(input_path), output_path)
            tqdm.write(f"  [ID] {filename} (MSP: {msp_score:.4f}) -> Linked in output")
        else:
            ood_count += 1
            tqdm.write(f"  [OOD] {filename} (MSP: {msp_score:.4f}) -> Filtered out")

    print("\n--- Filtering Complete ---")
    print(f"Total images processed: {len(image_files)}")
    print(f"In-distribution images (linked): {id_count}")
    print(f"Out-of-distribution images (filtered): {ood_count}")
    print(f"Results available in: {args.output_dir}")


if __name__ == "__main__":
    main()
