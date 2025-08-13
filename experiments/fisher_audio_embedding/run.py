#!/usr/bin/env python
import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
import sys
import os
import argparse
from pathlib import Path
import urllib.request

# This script assumes the 'models' directory from the FISHER repo is placed
# inside 'experiments/fisher_audio_embedding/'.
# We add the parent directory to the path to allow for relative imports.
sys.path.append(str(Path(__file__).resolve().parent))
from models.fisher import FISHER

MODEL_URL = "https://cloud.tsinghua.edu.cn/f/630a4b1b2962481a9150/?dl=1"
DEFAULT_MODEL_PATH = "fisher-tiny.pt"
DUMMY_WAV_PATH = "dummy_signal.wav"


def download_model(model_path):
    """Downloads the FISHER-tiny model if it doesn't exist."""
    if not os.path.exists(model_path):
        print(f"Downloading FISHER-tiny model to {model_path}...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("Download complete.")


def generate_dummy_audio(wav_path):
    """Generates a dummy WAV file for demonstration."""
    if os.path.exists(wav_path):
        return
    print(f"--- Generating a dummy audio file at: {wav_path} ---")
    sample_rate = 16000
    duration_s = 5
    frequency = 440
    num_samples = int(sample_rate * duration_s)
    time = torch.linspace(0.0, duration_s, num_samples)
    amplitude = int(torch.iinfo(torch.int16).max * 0.5)
    waveform = (amplitude * torch.sin(2 * torch.pi * frequency * time)).unsqueeze(0)
    torchaudio.save(wav_path, waveform.to(torch.int16), sample_rate)


def get_audio_embedding(wav_path, model_path):
    """
    Extracts audio embeddings from a WAV file using the FISHER model.
    This function encapsulates the preprocessing and inference logic
    from the original FISHER repository.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This experiment requires a GPU.")

    # 1. Load and preprocess audio
    print(f"Loading and preprocessing audio from: {wav_path}")
    wav, sr = torchaudio.load(wav_path)
    wav = wav.to("cuda")

    # Preprocessing steps from FISHER README
    wav = wav - wav.mean()
    stft = torchaudio.transforms.Spectrogram(
        n_fft=25 * sr // 1000,
        win_length=None,
        hop_length=10 * sr // 1000,
        power=1,
        center=False,
    ).to("cuda")
    spec = torch.log(torch.abs(stft(wav)) + 1e-10)
    spec = spec.transpose(-2, -1)  # [1, time, freq]

    # Normalization constants from FISHER README
    spec = (spec + 3.017344307886898) / (2.1531635155379805 * 2)
    print(f"Original spectrogram shape: {spec.shape}")

    # 2. Load model
    print(f"Loading FISHER model from: {model_path}")
    model = FISHER.from_pretrained(model_path)
    model = model.cuda()
    model.eval()

    # 3. Prepare tensor for model input
    # Time-wise cutoff
    if spec.shape[-2] > 1024:
        spec = spec[:, :1024]
    # Freq-wise padding
    if spec.shape[-1] < model.cfg.band_width:
        spec = F.pad(spec, (0, model.cfg.band_width - spec.shape[-1]))
    spec = spec.unsqueeze(1)  # Add channel dimension
    print(f"Final input tensor shape for model: {spec.shape}")

    # 4. Run inference
    print("Running inference to extract features...")
    with torch.no_grad():
        # Use autocast for mixed precision inference as recommended
        with torch.autocast("cuda"):
            representation = model.extract_features(spec)

    print("\n--- Inference Complete ---")
    print(f"Successfully extracted features.")
    print(f"Output representation shape: {representation.shape}")
    return representation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract audio embeddings using FISHER."
    )
    parser.add_argument(
        "--audio_file",
        type=str,
        default=DUMMY_WAV_PATH,
        help=f"Path to the input audio file. A dummy file will be generated if not provided.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the FISHER model checkpoint. Will be downloaded if it doesn't exist.",
    )
    args = parser.parse_args()

    # Get the directory of the current script
    script_dir = Path(__file__).resolve().parent
    model_path_abs = script_dir / args.model_path
    audio_file_abs = script_dir / args.audio_file

    # Prepare assets
    download_model(model_path_abs)
    generate_dummy_audio(audio_file_abs)

    # Run the main function
    get_audio_embedding(audio_file_abs, model_path_abs)
