import os
import argparse
import gdown
import cv2
from tqdm import tqdm
import glob

# The Google Drive folder ID containing the ISL dataset videos.
# Extracted from the URL in the source repo: https://drive.google.com/drive/folders/1EQfkP9LGNqL8WkwscS7TvQWAQgEjbJxG
GOOGLE_DRIVE_FOLDER_ID = "1EQfkP9LGNqL8WkwscS7TvQWAQgEjbJxG"


def extract_frames(video_path, output_dir, frame_skip=5):
    """
    Extracts frames from a single video file and saves them to a directory.

    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save the extracted frames.
        frame_skip (int): Save one frame every `frame_skip` frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = 0
    saved_frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            frame_filename = f"frame_{saved_frame_count:04d}.png"
            cv2.imwrite(os.path.join(output_dir, frame_filename), frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    # print(f"Extracted {saved_frame_count} frames from {os.path.basename(video_path)}")


def main(args):
    """
    Main function to download and process the ISL dataset.
    """
    # 1. Create output directories
    raw_video_dir = os.path.join(args.output_dir, "raw_videos")
    processed_frames_dir = os.path.join(args.output_dir, "processed_frames")
    os.makedirs(raw_video_dir, exist_ok=True)
    os.makedirs(processed_frames_dir, exist_ok=True)

    print(f"Output directory set to: {args.output_dir}")

    # 2. Download the dataset videos from Google Drive
    print(
        f"Downloading ISL video dataset from Google Drive folder ID: {GOOGLE_DRIVE_FOLDER_ID}..."
    )
    # Check if directory is already populated to avoid re-downloading
    if not os.listdir(raw_video_dir):
        gdown.download_folder(
            id=GOOGLE_DRIVE_FOLDER_ID,
            output=raw_video_dir,
            quiet=False,
            use_cookies=False,
        )
        print("Download complete.")
    else:
        print("Raw video directory is not empty. Skipping download.")

    # 3. Find all downloaded video files (might be nested one level deep)
    video_files = glob.glob(os.path.join(raw_video_dir, "*.mp4"))
    if not video_files:
        # Look in subdirectories if the folder download created a parent folder
        subdirs = [
            d
            for d in os.listdir(raw_video_dir)
            if os.path.isdir(os.path.join(raw_video_dir, d))
        ]
        if subdirs:
            video_files = glob.glob(os.path.join(raw_video_dir, subdirs[0], "*.mp4"))

    if not video_files:
        print("Error: No .mp4 video files found in the download directory.")
        return

    print(f"Found {len(video_files)} video files to process.")

    # 4. Process each video file
    for video_path in tqdm(video_files, desc="Processing videos"):
        # Extract class name from filename, e.g., 'class_0.mp4' -> 'class_0'
        class_name = os.path.splitext(os.path.basename(video_path))[0]
        class_output_dir = os.path.join(processed_frames_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        extract_frames(video_path, class_output_dir, frame_skip=args.frame_skip)

    print("\nDataset preparation complete.")
    print(f"Processed frames are saved in: {processed_frames_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and prepare the Indian Sign Language (ISL) dataset."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/isl_dataset",
        help="The directory where the dataset will be downloaded and processed.",
    )
    parser.add_argument(
        "--frame_skip",
        type=int,
        default=5,
        help="The interval at which to sample frames from the videos (e.g., 5 means save 1 every 5 frames).",
    )

    args = parser.parse_args()
    main(args)
