import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
import sys
import os
from pathlib import Path

# This path is configured in the accompanying Dockerfile to point to the cloned FISHER repo.
sys.path.append('/app/fisher')
from models.fisher import FISHER

class FisherAudioExtractor:
    """
    A wrapper for the FISHER model to extract audio embeddings.

    This class encapsulates the preprocessing and inference logic described
    in the FISHER repository for generating representations from audio signals.
    """
    def __init__(self, model_path: str, device: str = 'cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This model requires a GPU.")
        self.device = torch.device(device)

        print(f"Loading FISHER model from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
            
        self.model = FISHER.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("FISHER model loaded successfully.")

    def _preprocess_waveform(self, wav: torch.Tensor, sr: int):
        """
        Preprocesses a raw waveform into a spectrogram suitable for the FISHER model.
        This logic is taken directly from the FISHER README.
        """
        # 1. Denoise and center
        wav = wav - wav.mean()

        # 2. Compute STFT
        stft_transform = torchaudio.transforms.Spectrogram(
            n_fft=25 * sr // 1000,
            win_length=None,
            hop_length=10 * sr // 1000,
            power=1,
            center=False
        )
        spec = torch.log(torch.abs(stft_transform(wav)) + 1e-10)
        spec = spec.transpose(-2, -1)  # [1, time, freq]

        # 3. Normalize using pre-computed stats from the paper/repo
        spec = (spec + 3.017344307886898) / (2.1531635155379805 * 2)
        
        return spec

    def extract_features(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Extracts a feature embedding from an audio waveform.
        
        Args:
            wav (torch.Tensor): The input audio waveform.
            sr (int): The sample rate of the audio.
            
        Returns:
            torch.Tensor: The extracted feature representation on the CPU.
        """
        # Preprocess the audio to get a spectrogram
        spec = self._preprocess_waveform(wav, sr)

        # Apply model-specific padding and truncation
        # time-wise cutoff
        if spec.shape[-2] > 1024:
            spec = spec[:, :1024]
        # freq-wise padding
        if spec.shape[-1] < self.model.cfg.band_width:
            spec = F.pad(spec, (0, self.model.cfg.band_width - spec.shape[-1]))
        
        spec = spec.unsqueeze(1).to(self.device)
        print(f"Input spec shape for model: {spec.shape}")
        
        # Run inference
        with torch.no_grad():
            # Use autocast for mixed precision inference as recommended
            with torch.autocast('cuda'):
                representation = self.model.extract_features(spec)
        
        return representation.cpu()


def main():
    """
    Main execution function for the experiment.
    """
    print("--- Starting FISHER Audio Feature Extraction Experiment ---")
    
    # Define paths
    output_dir = Path("./experiment_outputs")
    output_dir.mkdir(exist_ok=True)
    wav_path = output_dir / "dummy_signal.wav"
    model_path = '/app/models/fisher-tiny.pt'

    # 1. Generate a dummy wav file for the demo
    print(f"--- Generating a dummy audio file at {wav_path} ---")
    sample_rate = 16000
    duration_s = 5
    frequency = 440
    num_samples = int(sample_rate * duration_s)
    time = torch.linspace(0., duration_s, num_samples)
    amplitude = int(torch.iinfo(torch.int16).max * 0.5)
    waveform = (amplitude * torch.sin(2 * torch.pi * frequency * time)).unsqueeze(0)
    torchaudio.save(str(wav_path), waveform.to(torch.int16), sample_rate)
    print("Dummy audio file generated.")

    # 2. Load the waveform
    wav, sr = torchaudio.load(str(wav_path))
    
    # 3. Initialize the extractor and get features
    try:
        extractor = FisherAudioExtractor(model_path=model_path)
        features = extractor.extract_features(wav, sr)
        
        print("\n--- Inference Complete ---")
        print("Successfully extracted audio features using FISHER.")
        print(f"Output representation shape: {features.shape}")
        
        # Simple validation
        assert len(features.shape) == 3, f"Expected 3 dimensions, but got {len(features.shape)}"
        assert features.shape[0] == 1, f"Expected batch size of 1, but got {features.shape[0]}"
        print("\n✅ Success: Output shape is valid.")

    except Exception as e:
        print(f"\n❌ ERROR: An exception occurred during the experiment: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
