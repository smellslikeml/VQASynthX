import torch
import torchaudio
import argparse

def preprocess_audio(audio_path: str):
    """
    Applies the FISHER preprocessing pipeline to an audio file.

    This pipeline converts a raw audio waveform into a normalized
    log-spectrogram tensor, suitable for input to a transformer model.
    The steps are adapted directly from the FISHER project's inference code.

    Args:
        audio_path: Path to the input .wav file.

    Returns:
        A preprocessed spectrogram tensor.
    """
    # 1. Load the audio signal
    # You can replace it with your custom loading function for other signals
    try:
        wav, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"Error loading audio file: {audio_path}")
        raise e

    # 2. Pre-emphasis / Normalization
    wav = wav - wav.mean()

    # 3. Compute STFT Spectrogram
    # Parameters from FISHER README: 25ms window, 10ms hop
    stft_transform = torchaudio.transforms.Spectrogram(
        n_fft=25 * sr // 1000,
        win_length=None,  # defaults to n_fft
        hop_length=10 * sr // 1000,
        power=1,  # For magnitude spectrogram
        center=False
    )
    spec = stft_transform(wav)

    # 4. Logarithmic scaling
    spec = torch.log(torch.abs(spec) + 1e-10)

    # 5. Transpose to (Batch, Time, Freq)
    spec = spec.transpose(-2, -1)

    # 6. Normalize the spectrogram
    # These magic numbers are the mean/std from the FISHER training set
    mean = 3.017344307886898
    std = 2.1531635155379805
    spec = (spec + mean) / (std * 2)

    print(f"Successfully processed {audio_path}")
    print(f"Original sample rate: {sr} Hz")
    print(f"Output tensor shape: {spec.shape}")

    return spec

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess an audio file using the FISHER methodology."
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="Path to the input audio (.wav) file."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output tensor (.pt file)."
    )
    args = parser.parse_args()

    processed_tensor = preprocess_audio(args.audio_path)
    torch.save(processed_tensor, args.output_path)
    print(f"Preprocessed tensor saved to {args.output_path}")
