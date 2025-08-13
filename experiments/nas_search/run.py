"""
Proof-of-Concept: Neural Architecture Search (NAS) for VQASynth.

This script integrates the `ConfigurableOptimizer` library to demonstrate how
Neural Architecture Search (NAS) can be applied within the VQASynth experimental
framework. It runs a minimal DARTS-based search on the CIFAR-10 dataset.

The goal is to establish a foundation for future experiments that could use NAS
to discover more efficient, specialized models for components of the VQASynth
data generation pipeline (e.g., a lightweight depth estimation backbone or
a scene classifier).

This experiment is self-contained and uses the default settings from
`ConfigurableOptimizer`'s light demo.
"""

from confopt.profile import DARTSProfile
from confopt.train import Experiment
from confopt.enums import TrainerPresetType, SearchSpaceType, DatasetType


def run_nas_experiment():
    """Initializes and runs the NAS experiment."""
    print("--- Starting Neural Architecture Search Experiment ---")
    print(
        "This POC uses ConfigurableOptimizer to find an efficient architecture on CIFAR-10."
    )
    print("The long-term vision is to adapt this for optimizing VQASynth components.")

    # Define the search profile using a preset for DARTS.
    # We limit it to 3 epochs for a quick demonstration.
    profile = DARTSProfile(trainer_preset=TrainerPresetType.DARTS, epochs=3)

    # Define the experiment, specifying the search space and dataset.
    # The DARTS search space is a common benchmark for cell-based architecture search.
    # CIFAR-10 is used as the standard dataset for this demonstration.
    experiment = Experiment(
        search_space=SearchSpaceType.DARTS,
        dataset=DatasetType.CIFAR10,
        data_path="./data",  # Specify a local data path for datasets
    )

    # Run the supernet training, which is the core of the one-shot NAS process.
    print("\nStarting supernet training...")
    experiment.train_supernet(profile)
    print("\n--- Supernet training finished successfully. ---")
    print(
        "Experiment complete. This demonstrates the successful integration of the NAS framework."
    )


if __name__ == "__main__":
    run_nas_experiment()
