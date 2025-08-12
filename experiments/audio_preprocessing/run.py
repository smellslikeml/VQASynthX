import torch
import torchaudio
import os
from pathlib import Path

# --- Constants from FISHER README --- 
# These constants are derived from the training set used for FISHER
# and are crucial for correct feature normalization.
NORM_ADD_CONSTANT = 3.017344307886898
NORM_DIV_CONSTANT = 2.1531635155379805 * 2

def generate_dummy_audio(path: Path, sample_rate: int = 16000, duration_s: int = 5, frequency: int = 440):
    """Generates a simple sine wave audio file for demonstration."""
    print(f"Generating dummy audio file at: {path}")
    num_samples = int(sample_rate * duration_s)
    time = torch.linspace(0., duration_s, num_samples)
    amplitude = int(torch.iinfo(torch.int16).max * 0.5)
    waveform = (amplitude * torch.sin(2 * torch.pi * frequency * time)).unsqueeze(0)
    torchaudio.save(path, waveform.to(torch.int16), sample_rate)
    print("Dummy audio generated.")

def preprocess_audio_fisher(wav_path: Path, output_path: Path):
    """
    Preprocesses an audio file using the method from the FISHER repository.
    This involves creating a normalized log-magnitude spectrogram.
    """
    print(f"\nProcessing audio file: {wav_path}")
    try:
        wav, sr = torchaudio.load(wav_path)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    # 1. Center the waveform
    wav = wav - wav.mean()

    # 2. Compute STFT (Spectrogram)
    # Parameters from FISHER README: n_fft=25ms, hop_length=10ms
    stft_transform = torchaudio.transforms.Spectrogram(
        n_fft=int(25 * sr / 1000),
        win_length=None,  # defaults to n_fft
        hop_length=int(10 * sr / 1000),
        power=1,  # For magnitude spectrogram, not power
        center=False,
    )
    spec = stft_transform(wav)

    # 3. Apply log transformation for stability
    spec = torch.log(torch.abs(spec) + 1e-10)

    # 4. Transpose to [Batch, Time, Freq] format, which is common for transformers
    spec = spec.transpose(-2, -1)

    # 5. Normalize using pre-calculated stats from FISHER's training data
    spec = (spec + NORM_ADD_CONSTANT) / NORM_DIV_CONSTANT

    print(f"Spectrogram generated with shape: {spec.shape}")

    # 6. Save the preprocessed tensor
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(spec, output_path)
    print(f"Preprocessed spectrogram saved to: {output_path}")

def main():
    """
    Main function to set up directories, generate dummy data,
    and run the preprocessing experiment.
    """
    print("--- Starting FISHER-inspired Audio Preprocessing Experiment ---")
    
    # Setup directories
    base_dir = Path("./exp_data")
    output_dir = base_dir / "audio_features"
    input_dir = base_dir / "audio_input"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    # Generate a dummy input file
    dummy_wav_path = input_dir / "sample_audio.wav"
    generate_dummy_audio(dummy_wav_path)

    # Define the output path for the processed tensor
    output_tensor_path = output_dir / f"{dummy_wav_path.stem}_spectrogram.pt"

    # Run the preprocessing
    preprocess_audio_fisher(dummy_wav_path, output_tensor_path)
    
    # Verification step
    if output_tensor_path.exists():
        print("\n--- Verification ---")
        loaded_spec = torch.load(output_tensor_path)
        print(f"Successfully loaded saved tensor from {output_tensor_path}")
        print(f"Tensor shape: {loaded_spec.shape}")
        print(f"Tensor dtype: {loaded_spec.dtype}")
        print("\nExperiment finished successfully.")
    else:
        print("\n--- Error ---")
        print("Output file was not created. Experiment failed.")

if __name__ == "__main__":
    main()
