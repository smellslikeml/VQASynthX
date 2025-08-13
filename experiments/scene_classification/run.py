import argparse
import os
import json
import sys

import torch
import torch.utils.data
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# This script assumes that the GFNet-Dynn repository has been cloned
# and its root directory added to the PYTHONPATH.
# Example: export PYTHONPATH=$PYTHONPATH:/path/to/GFNet-Dynn
from models.gfnet_dynn import GFNet_dynn


def build_inference_transform(img_size):
    """Creates the specific transformation pipeline for inference."""
    t = []
    # In the source repo, they resize to 256 for a 224 input.
    # We maintain the same ratio.
    size = int((256 / 224) * img_size)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC)
    )
    t.append(transforms.CenterCrop(img_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def main(args):
    """Main inference script."""
    device = torch.device(args.device)

    # --- Model Preparation ---
    print(f"Loading model: {args.model_name}")
    # Note: The 'add_output' parameter must match the number of early exits
    # in the pre-trained model checkpoint.
    model = GFNet_dynn(
        num_classes=args.num_classes,
        model_name=args.model_name,
        add_output=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    )

    if not os.path.exists(args.model_checkpoint):
        print(f"Error: Checkpoint file not found at {args.model_checkpoint}")
        sys.exit(1)

    print(f"Loading checkpoint from {args.model_checkpoint}")
    checkpoint = torch.load(args.model_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    # --- Data Preparation ---
    transform = build_inference_transform(args.img_size)
    dataset = datasets.ImageFolder(args.image_dir, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    class_names = dataset.classes
    print(f"Found {len(dataset)} images in {len(class_names)} classes.")

    # --- Inference ---
    results = {}
    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            images = images.to(device, non_blocking=True)

            # We use the final exit for the highest accuracy prediction
            output = model(images)[-1]

            _, preds = torch.max(output, 1)
            preds_cpu = preds.cpu().numpy()

            # Map predictions to class names and store with original file paths
            start_index = i * args.batch_size
            for j, p_idx in enumerate(preds_cpu):
                original_path = dataset.samples[start_index + j][0]
                filename = os.path.basename(original_path)
                results[filename] = class_names[p_idx]

    # --- Save Results ---
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Classification complete. Results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GFNet-Dynn Inference Script")
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing images to classify.",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=True,
        help="Path to the pre-trained model checkpoint.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="classification_results.json",
        help="Path to save the output JSON file.",
    )

    # Model parameters (must match the checkpoint)
    parser.add_argument(
        "--model_name",
        type=str,
        default="gfnet_h_ti",
        help="Name of the GFNet model architecture.",
    )
    parser.add_argument(
        "--num_classes", type=int, default=10, help="Number of classes for the model."
    )
    parser.add_argument("--img_size", type=int, default=224, help="Input image size.")

    # Dataloader and device parameters
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for inference."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers."
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference (cuda/cpu).",
    )

    args = parser.parse_args()
    main(args)
