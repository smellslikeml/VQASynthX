import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
import sys
import argparse
import os
from models.fisher import FISHER


def process_audio(input_path: str, output_path: str):
    """
    Processes a single audio file using the FISHER model to generate an embedding.
    """
    print(f"--- Starting FISHER Audio Processing ---")
    print(f"Input audio file: {input_path}")
    print(f"Output embedding path: {output_path}")

    if not torch.cuda.is_available():
        print(
            "ERROR: CUDA is not available. This stage requires a GPU.", file=sys.stderr
        )
        sys.exit(1)

    # 1. Load and preprocess audio signal
    try:
        wav, sr = torchaudio.load(input_path)
    except Exception as e:
        print(
            f"ERROR: Could not load audio file {input_path}. Reason: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Preprocessing from FISHER README
    wav = wav - wav.mean()
    # Ensure STFT parameters are integers
    n_fft = int(25 * sr // 1000)
    hop_length = int(10 * sr // 1000)

    stft_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        win_length=None,  # defaults to n_fft
        hop_length=hop_length,
        power=1,
        center=False,
    )
    spec = torch.log(torch.abs(stft_transform(wav)) + 1e-10)
    spec = spec.transpose(-2, -1)  # [Channel, Time, Freq]

    # Normalization constants from README
    spec = (spec + 3.017344307886898) / (2.1531635155379805 * 2)

    # 2. Load pre-trained FISHER model
    model_path = "models/fisher-tiny.pt"
    print(f"Loading model from: {model_path}")
    model = FISHER.from_pretrained(model_path)
    model = model.cuda()
    model.eval()
    print("Model loaded successfully.")

    # 3. Prepare tensor for model input
    # time-wise cutoff
    if spec.shape[-2] > 1024:
        spec = spec[:, :1024]

    # freq-wise padding
    band_width = model.cfg.band_width
    if spec.shape[-1] < band_width:
        spec = F.pad(spec, (0, band_width - spec.shape[-1]))

    # Model expects [B, C, Time, Freq].
    # torchaudio gives [C,T] -> spec [C,Time,Freq]. Add Batch dim.
    if spec.dim() == 3:
        spec = spec.unsqueeze(0).cuda()  # [1, C, Time, Freq]
    else:
        print(f"ERROR: Unexpected spectrogram dimension: {spec.dim()}", file=sys.stderr)
        sys.exit(1)

    print(f"Input spec shape for model: {spec.shape}")

    # 4. Run inference
    with torch.no_grad():
        with torch.autocast("cuda"):
            representation = model.extract_features(spec)

    print("\n--- Inference Complete ---")
    print(f"Successfully extracted features.")
    print(f"Output representation shape: {representation.shape}")

    # 5. Save the output embedding
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(representation.cpu(), output_path)
    print(f"Embedding saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate audio embeddings using FISHER model."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input audio file (.wav).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output embedding tensor (.pt).",
    )
    args = parser.parse_args()

    process_audio(args.input_path, args.output_path)
