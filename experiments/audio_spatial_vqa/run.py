import torch
import torchaudio
import os
import urllib.request

# This import will work after installing from requirements.txt
from matpac.model import get_matpac

print("--- MATPAC Audio Feature Extraction for VQASynth PoC ---")

# 1. Define constants
MODEL_URL = "https://github.com/aurianworld/matpac/releases/download/Initial_release/matpac_10_2048.pt"
CKPT_PATH = "matpac_10_2048.pt"
SAMPLE_RATE = 16000
AUDIO_FILE = "test_audio.wav"
DURATION_S = 5
FREQUENCY = 440  # A4 note


def download_model(url, path):
    """Downloads the model checkpoint if it doesn't exist."""
    if not os.path.exists(path):
        print(f"Downloading MATPAC model from {url}...")
        try:
            urllib.request.urlretrieve(url, path)
            print(f"Model saved to {path}")
        except Exception as e:
            print(f"Error downloading model: {e}")
            exit(1)
    else:
        print(f"Model checkpoint already exists at {path}")


def generate_dummy_audio(file_path, duration_s, sample_rate, freq):
    """Generates a simple sine wave audio file."""
    print(f"Generating a dummy audio file: {file_path}...")
    num_samples = duration_s * sample_rate
    time = torch.linspace(0.0, duration_s, num_samples)
    # Generate a mono waveform in the [-1, 1] range as required by MATPAC
    waveform = (0.5 * torch.sin(2 * torch.pi * freq * time)).unsqueeze(0)
    torchaudio.save(file_path, waveform, sample_rate)
    print("Dummy audio file created.")


def main():
    # Download the pre-trained model checkpoint
    download_model(MODEL_URL, CKPT_PATH)

    # Generate a dummy audio file for demonstration
    generate_dummy_audio(AUDIO_FILE, DURATION_S, SAMPLE_RATE, FREQUENCY)

    # Load the audio
    print(f"Loading audio from {AUDIO_FILE}...")
    x, sr = torchaudio.load(AUDIO_FILE)

    # VQASynth processes data in batches, so we ensure the audio tensor has a batch dimension
    # MATPAC expects (bs, n_samples)
    if len(x.shape) == 1:
        x = x.unsqueeze(0)

    print(f"Audio loaded. Shape: {x.shape}, Sample Rate: {sr}")

    # Instantiate the MATPAC model and move to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading MATPAC model from {CKPT_PATH}...")
    model = get_matpac(checkpoint_path=CKPT_PATH)
    model.to(device)
    x = x.to(device)
    model.eval()  # Set model to evaluation mode
    print("Model loaded successfully.")

    # Extract features from the audio
    print("Extracting features...")
    with torch.no_grad():
        # emb is the final, time-pooled embedding
        # layer_results contains embeddings from all 12 transformer layers
        emb, layer_results = model(x)
    print("Feature extraction complete.")

    # Print results
    print(f"\n--- Results ---")
    print(f"Output embedding shape (time-pooled): {emb.shape}")
    print(f"Layer results shape (all layers): {layer_results.shape}")
    print(f"Example embedding values (first 5): {emb.flatten()[:5].cpu().numpy()}")

    print("\n--- VQASynth Integration Note ---")
    print(
        "This audio embedding could be concatenated with visual features (e.g., from CLIP)"
    )
    print("to provide multimodal context for generating spatial VQA pairs.")
    print("\n--- PoC Finished Successfully ---")


if __name__ == "__main__":
    main()
