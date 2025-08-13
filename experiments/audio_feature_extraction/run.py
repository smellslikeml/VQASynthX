#!/usr/bin/env python
import torch
import torchaudio
import torch.nn.functional as F
import sys
import os
from pathlib import Path

# --- Setup: Assumed Directory Structure ---
# This script assumes the following directory structure for testing:
# /app/
# ├── experimental-vqasynth/      (TARGET repo)
# │   └── experiments/
# │       └── audio_feature_extraction/
# │           └── run.py
# ├── FISHER/                       (SOURCE repo)
# │   ├── models/
# │   │   ├── fisher.py
# │   │   └── ...
# │   └── ...
# └── models/
#     └── fisher-tiny.pt          (Downloaded checkpoint)
#
# We add the FISHER directory to the path to allow direct imports.
# This avoids vendoring the entire model code into the VQASynth repo for this initial experiment.

try:
    # Adjust path to find the FISHER model code
    fisher_repo_path = Path(__file__).parent.parent.parent.parent / "FISHER"
    if not fisher_repo_path.exists():
        raise ImportError(
            "FISHER repository not found. Please clone it next to the 'experimental-vqasynth' repo."
        )
    sys.path.append(str(fisher_repo_path))
    from models.fisher import FISHER
except ImportError as e:
    print(f"Error: {e}", file=sys.stderr)
    print(
        "Could not import FISHER model. Please ensure the FISHER repository is cloned at the same level as 'experimental-vqasynth'.",
        file=sys.stderr,
    )
    sys.exit(1)


def generate_dummy_audio(
    save_path: str = "dummy_signal.wav", sample_rate: int = 16000
) -> None:
    """Generates a simple sine wave audio file for demonstration."""
    print(f"--- Generating a dummy audio file at {save_path} ---")
    duration_s = 5
    frequency = 440
    num_samples = int(sample_rate * duration_s)
    time = torch.linspace(0.0, duration_s, num_samples)
    amplitude = int(torch.iinfo(torch.int16).max * 0.5)
    waveform = (amplitude * torch.sin(2 * torch.pi * frequency * time)).unsqueeze(0)
    torchaudio.save(save_path, waveform.to(torch.int16), sample_rate)
    print("Dummy audio file generated.")


def preprocess_audio_for_fisher(
    waveform: torch.Tensor, sample_rate: int
) -> torch.Tensor:
    """
    Preprocesses a raw audio waveform into a spectrogram tensor as required by FISHER.
    This logic is adapted directly from the FISHER repository's README.
    """
    print("--- Preprocessing audio waveform for FISHER ---")
    # 1. Mean subtraction
    waveform = waveform - waveform.mean()

    # 2. STFT
    stft_transform = torchaudio.transforms.Spectrogram(
        n_fft=25 * sample_rate // 1000,
        win_length=None,
        hop_length=10 * sample_rate // 1000,
        power=1,
        center=False,
    )
    spec = torch.abs(stft_transform(waveform))

    # 3. Log scaling
    spec = torch.log(spec + 1e-10)

    # 4. Transpose to [Batch, Time, Freq]
    spec = spec.transpose(-2, -1)

    # 5. Normalize using pre-calculated stats from FISHER
    spec = (spec + 3.017344307886898) / (2.1531635155379805 * 2)

    print(f"Preprocessed spectrogram shape: {spec.shape}")
    return spec


def extract_fisher_features(
    model_path: str, audio_path: str, device: str = "cuda"
) -> torch.Tensor:
    """
    Main function to load the FISHER model, process an audio file, and extract features.
    """
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but this experiment requires a GPU.")

    # 1. Load model
    print(f"--- Loading FISHER model from: {model_path} ---")
    model = FISHER.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # 2. Load and preprocess audio
    waveform, sr = torchaudio.load(audio_path)
    spec = preprocess_audio_for_fisher(waveform, sr)

    # 3. Prepare tensor for model input (padding, etc.)
    # time-wise cutoff
    if spec.shape[-2] > 1024:
        spec = spec[:, :1024]
    # freq-wise padding
    if spec.shape[-1] < model.cfg.band_width:
        spec = F.pad(spec, (0, model.cfg.band_width - spec.shape[-1]))

    # Add channel and batch dimension, move to device
    spec = spec.unsqueeze(1).to(device)
    print(f"Final input tensor shape for model: {spec.shape}")

    # 4. Run inference
    print("--- Running inference to extract features ---")
    with torch.no_grad():
        # Use autocast for mixed precision as recommended
        with torch.autocast(device):
            representation = model.extract_features(spec)

    print("\n--- Inference Complete ---")
    print(f"Successfully extracted features.")
    print(f"Output representation shape: {representation.shape}")

    return representation


if __name__ == "__main__":
    # Define paths based on the assumed directory structure
    repo_root = Path(__file__).parent.parent.parent
    # In a container, the layout is /app/experimental-vqasynth, /app/FISHER, /app/models/fisher-tiny.pt
    model_dir = repo_root.parent / "models"
    model_checkpoint_path = model_dir / "fisher-tiny.pt"
    dummy_audio_path = repo_root / "dummy_signal.wav"

    # Check if checkpoint exists
    if not model_checkpoint_path.exists():
        print(
            f"ERROR: Model checkpoint not found at '{model_checkpoint_path}'",
            file=sys.stderr,
        )
        print(
            "Please download it using the link in the FISHER README and place it in a 'models' directory next to the repos.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Generate a dummy file to ensure the script can run
    generate_dummy_audio(str(dummy_audio_path))

    # Run the main feature extraction process
    audio_features = extract_fisher_features(
        model_path=str(model_checkpoint_path),
        audio_path=str(dummy_audio_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Clean up the dummy file
    os.remove(dummy_audio_path)
    print(f"\nCleaned up {dummy_audio_path}.")
