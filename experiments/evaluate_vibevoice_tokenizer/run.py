import torch
import librosa
import soundfile as sf
import numpy as np
import os

# NOTE: This script requires the 'vibevoice' package to be installed.
# pip install git+https://github.com/microsoft/VibeVoice.git
from vibevoice.modular.modeling_vibevoice import VibeVoiceModel
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor


def create_sample_wav(path="sample.wav", sr=24000, duration=3, freq=440.0):
    """Creates a simple sine wave audio file for testing if it doesn't exist."""
    if os.path.exists(path):
        return
    print(f"Creating sample audio file at '{path}'...")
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    amplitude = np.iinfo(np.int16).max * 0.5
    data = (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.int16)
    sf.write(path, data, sr)
    print("Sample audio file created.")


def main():
    """
    This script evaluates the VibeVoice acoustic tokenizer, a core component
    of its efficient speech synthesis pipeline. The goal is to assess its
    utility for compressing and reconstructing audio for potential use in
    the VQASynth data generation pipeline, extending it to the audio modality.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if "cuda" in device else torch.float32
    print(f"Using device: {device} with dtype: {dtype}")

    # 1. Load VibeVoice model and its associated processor
    # We load the full model to access its pre-trained tokenizer component.
    model_path = "microsoft/VibeVoice-1.5B"
    print(f"Loading model and processor from '{model_path}'...")
    model = VibeVoiceModel.from_pretrained(model_path, torch_dtype=dtype).to(device)
    processor = VibeVoiceProcessor.from_pretrained(model_path)
    model.eval()

    # 2. Prepare input audio
    # The script creates a sample file to be self-contained.
    target_sr = processor.sampling_rate
    sample_path = "sample.wav"
    output_path = "output_reconstructed.wav"
    create_sample_wav(sample_path, sr=target_sr)

    print(f"Loading audio from '{sample_path}'...")
    waveform, sr = librosa.load(sample_path, sr=target_sr)
    waveform_tensor = (
        torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(device)
    )

    if waveform_tensor.dtype != model.dtype:
        waveform_tensor = waveform_tensor.to(model.dtype)

    # 3. Encode audio into discrete tokens using the acoustic tokenizer
    acoustic_tokenizer = model.acoustic_tokenizer
    print("Encoding audio waveform to acoustic tokens...")
    with torch.no_grad():
        # The .encode() method of the tokenizer compresses the waveform into token IDs.
        codes, _ = acoustic_tokenizer.encode(waveform_tensor)

    print(f"Original waveform shape: {waveform_tensor.shape}")
    print(f"Encoded token shape: {codes.shape} (Batch, Codebooks, Time)")

    # 4. Decode tokens back into a waveform
    print("Decoding acoustic tokens back to waveform...")
    with torch.no_grad():
        reconstructed_waveform_tensor = acoustic_tokenizer.decode(codes)

    print(f"Reconstructed waveform shape: {reconstructed_waveform_tensor.shape}")

    # 5. Save output and report results
    reconstructed_waveform = (
        reconstructed_waveform_tensor.squeeze().cpu().to(torch.float32).numpy()
    )
    print(f"Saving reconstructed audio to '{output_path}'...")
    sf.write(output_path, reconstructed_waveform, target_sr)

    original_elements = waveform_tensor.numel()
    token_elements = codes.numel()
    print("\n--- Experiment Summary ---")
    print(f"Audio processed and saved to '{output_path}'.")
    print(f"Input elements (waveform): {original_elements}")
    print(f"Compressed elements (tokens): {token_elements}")
    print(
        f"Compression factor (waveform elements / token elements): {original_elements / token_elements:.2f}x"
    )
    print(
        "Success: The script completed. Listen to the output file to qualitatively assess reconstruction quality."
    )


if __name__ == "__main__":
    main()
