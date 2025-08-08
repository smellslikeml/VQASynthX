import argparse
import os
import glob
import numpy as np
import torch
from PIL import Image
from pytorch_msssim import ms_ssim

def load_depth_map(path):
    """Loads a depth map from file and normalizes it to [0, 1]."""
    # Assuming depth maps are saved as 16-bit PNGs or similar
    with Image.open(path) as img:
        depth = np.array(img, dtype=np.float32)
    
    if depth.max() > 0:
        depth /= depth.max() # Normalize to [0, 1] for MS-SSIM
    
    return torch.from_numpy(depth).unsqueeze(0).unsqueeze(0) # (B, C, H, W) -> (1, 1, H, W)

def main(args):
    """
    Main function to run the evaluation.
    This script is inspired by the evaluation methods in the tenvoo repository,
    adapting 3D medical image evaluation concepts to 2.5D depth map quality assessment.
    """
    print("Starting Ravine Evaluation...")
    print(f"Generated data path: {args.generated_path}")
    print(f"Ground truth path: {args.ground_truth_path}")

    generated_files = sorted(glob.glob(os.path.join(args.generated_path, '*.png')))
    gt_files = sorted(glob.glob(os.path.join(args.ground_truth_path, '*.png')))

    if not generated_files or not gt_files:
        print("Error: No image files found in one or both directories. Please check paths.")
        return

    # A more robust implementation would match by filename
    if len(generated_files) != len(gt_files):
        print(f"Warning: Mismatch in file counts. Generated: {len(generated_files)}, GT: {len(gt_files)}. Evaluating common subset.")
        min_len = min(len(generated_files), len(gt_files))
        generated_files = generated_files[:min_len]
        gt_files = gt_files[:min_len]

    total_ms_ssim = 0.0
    total_l1_error = 0.0
    evaluated_count = 0

    for gen_path, gt_path in zip(generated_files, gt_files):
        try:
            gen_depth_tensor = load_depth_map(gen_path)
            gt_depth_tensor = load_depth_map(gt_path)

            if gen_depth_tensor.shape != gt_depth_tensor.shape:
                print(f"Skipping {os.path.basename(gen_path)} due to shape mismatch.")
                continue

            # Calculate MS-SSIM, requires normalized tensors in [0, 1]
            # data_range should be 1.0 since we normalized the tensors.
            current_ms_ssim = ms_ssim(gen_depth_tensor, gt_depth_tensor, data_range=1.0, size_average=True)
            total_ms_ssim += current_ms_ssim.item()

            # Calculate Mean Absolute Error (L1)
            current_l1_error = torch.abs(gen_depth_tensor - gt_depth_tensor).mean()
            total_l1_error += current_l1_error.item()
            
            evaluated_count += 1
        except Exception as e:
            print(f"Could not process file pair: {gen_path}, {gt_path}. Error: {e}")

    if evaluated_count > 0:
        avg_ms_ssim = total_ms_ssim / evaluated_count
        avg_l1_error = total_l1_error / evaluated_count
        
        print("\n--- Evaluation Results ---")
        print(f"Evaluated pairs: {evaluated_count}")
        print(f"Average MS-SSIM: {avg_ms_ssim:.4f}")
        print(f"Average L1 Error: {avg_l1_error:.4f}")
        print("--------------------------")
    else:
        print("Evaluation failed. No files were processed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ravine Evaluator for VQASynth Depth Maps.")
    parser.add_argument('--generated_path', type=str, required=True, help='Path to the directory with generated depth maps.')
    parser.add_argument('--ground_truth_path', type=str, required=True, help='Path to the directory with ground truth depth maps.')
    
    args = parser.parse_args()
    main(args)
