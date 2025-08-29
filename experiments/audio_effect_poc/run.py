import torch
import torch.nn as nn
from torch import Tensor
from typing import List
import os
import numpy as np
import soundfile as sf

# Note: This experiment requires `neutone_sdk` and `soundfile` to be installed.
# pip install neutone_sdk soundfile

from neutone_sdk.neutone_sdk import WaveformToWaveformBase, export


# 1. Define the core PyTorch model
# This is a simple clipping effect, directly inspired by the Neutone SDK example.
class ClipperModel(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """Simple hard clipping between -0.5 and 0.5."""
        return torch.clip(x, min=-0.5, max=0.5)


# 2. Create the Neutone SDK Wrapper
# This class provides metadata and boilerplate for the Neutone host plugin.
class ClipperModelWrapper(WaveformToWaveformBase):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    @torch.jit.export
    def is_input_mono(self) -> bool:
        return False  # Expects stereo input

    @torch.jit.export
    def is_output_mono(self) -> bool:
        return False  # Outputs stereo

    @torch.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return []  # An empty list signifies support for all sample rates

    @torch.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        return []  # An empty list signifies support for all buffer sizes


# 3. Main execution block to export and test the model
def main():
    """
    Exports the model using Neutone SDK and runs a simple test to verify it.
    """
    print(">>> Initializing Clipper model and wrapper...")
    model = ClipperModel()
    model_wrapper = ClipperModelWrapper(model)

    output_dir = "exported_neutone_model"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.nm")

    print(f">>> Exporting model to {model_path}...")
    export(
        model_wrapper,
        path=model_path,
        description="A simple hard-clipping audio effect for the VQASynth pipeline.",
        short_name="VQASynth Clipper",
    )
    print(">>> Export successful.")

    # 4. Verification Step: Test the exported model on a dummy audio signal
    print("\n>>> Running verification test...")

    # Generate a sine wave that exceeds the clipping threshold
    sample_rate = 44100
    duration_s = 2
    frequency = 440  # A4 note
    t = np.linspace(0.0, duration_s, int(sample_rate * duration_s), endpoint=False)
    # Amplitude of 0.8, which will be clipped by the model to 0.5
    amplitude = 0.8
    # Create a stereo signal and shape it to (num_samples, 2) for soundfile
    input_audio_np = np.asarray(
        [amplitude * np.sin(2 * np.pi * frequency * t)] * 2, dtype=np.float32
    ).T

    input_wav_path = os.path.join(output_dir, "test_input.wav")
    output_wav_path = os.path.join(output_dir, "test_output.wav")
    sf.write(input_wav_path, input_audio_np, sample_rate)
    print(f"  - Generated test input: {input_wav_path}")

    # Load the exported model
    print("  - Loading exported Neutone model for inference...")
    loaded_model = torch.jit.load(model_path)

    # Prepare tensor for model input (needs to be [channels, samples])
    input_tensor = torch.from_numpy(input_audio_np.T)

    # Run inference
    print("  - Applying audio effect...")
    output_tensor = loaded_model.model(input_tensor)

    # Shape output back to (num_samples, 2) for soundfile
    output_audio_np = output_tensor.detach().numpy().T

    # Save the processed audio
    sf.write(output_wav_path, output_audio_np, sample_rate)
    print(f"  - Saved processed output: {output_wav_path}")
    print(
        "\n>>> Verification complete. Check the output WAV file to hear the clipping effect."
    )


if __name__ == "__main__":
    main()
