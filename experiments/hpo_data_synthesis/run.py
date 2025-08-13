import argparse
import logging
from pathlib import Path

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace.configuration_space import Configuration

# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_configspace() -> CS.ConfigurationSpace:
    """
    Defines the hyperparameter space for the VQASynth pipeline.
    This is inspired by the configurable nature of experiments in the confopt library.
    """
    cs = CS.ConfigurationSpace(seed=1234)

    # --- Pipeline Configuration ---
    # Choice of depth estimation model, as mentioned in VQASynth README
    depth_model_name = CSH.CategoricalHyperparameter(
        "depth_model_name",
        choices=["VGGT-Large", "DepthAnything", "DepthPro"],
        default_value="VGGT-Large",
    )

    # --- Filtering Stage Parameters ---
    min_box_area = CSH.UniformFloatHyperparameter(
        "min_box_area", lower=0.001, upper=0.05, default_value=0.01, log=True
    )

    # --- Prompting Stage Parameters ---
    num_questions_per_image = CSH.UniformIntegerHyperparameter(
        "num_questions_per_image", lower=5, upper=50, default_value=10
    )

    # --- Reasoning Stage Parameters ---
    use_cot_reasoning = CSH.CategoricalHyperparameter(
        "use_cot_reasoning", choices=[True, False], default_value=True
    )

    cs.add_hyperparameters(
        [
            depth_model_name,
            min_box_area,
            num_questions_per_image,
            use_cot_reasoning,
        ]
    )
    return cs


def run_synthesis_pipeline(config: Configuration, dataset_path: str, output_dir: str):
    """
    Simulates running the full VQASynth data generation pipeline with a given configuration.
    In a real implementation, this function would orchestrate the Docker stages.
    """
    logging.info("Starting VQASynth pipeline with configuration:")
    for key, value in config.items():
        logging.info(f"  - {key}: {value}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Stage 1: Depth Estimation
    logging.info(
        f"Running Depth Stage using model: {config['depth_model_name']} on dataset: {dataset_path}"
    )
    # Command: docker run ... depth_stage ... --model_name {config['depth_model_name']}

    # Stage 2: Scene Fusion
    logging.info("Running Scene Fusion Stage...")
    # Command: docker run ... scene_fusion_stage ...

    # Stage 3: Filtering
    logging.info(
        f"Running Filter Stage with min_box_area: {config['min_box_area']:.4f}"
    )
    # Command: docker run ... filter_stage ... --min_box_area {config['min_box_area']}

    # ... other stages would follow ...

    # Stage 7: Prompt Generation
    logging.info(
        f"Running Prompt Stage, generating {config['num_questions_per_image']} questions per image."
    )
    logging.info(
        f"Chain-of-Thought (CoT) reasoning enabled: {config['use_cot_reasoning']}"
    )
    # Command: docker run ... prompt_stage ... --num_questions {config['num_questions_per_image']}

    logging.info(f"Pipeline finished. Synthetic data saved to {output_dir}")
    # Write a dummy output file to confirm execution
    with open(Path(output_dir) / "synthetic_data.jsonl", "w") as f:
        f.write(
            '{"status": "completed", "config": ' + str(config.get_dictionary()) + "}\n"
        )


def main():
    """Main entry point to run a configured VQASynth experiment."""
    parser = argparse.ArgumentParser(
        description="A configurable runner for VQASynth data synthesis experiments, "
        "inspired by the methodology of the ConfigurableOptimizer project."
    )
    cs = get_configspace()

    # Add arguments from the config space
    for hp in cs.get_hyperparameters():
        # A small hack to handle boolean arguments correctly
        action = (
            "store_true"
            if isinstance(hp, CSH.CategoricalHyperparameter)
            and hp.choices == (False, True)
            else "store"
        )
        if action == "store_true":
            parser.add_argument(f"--{hp.name}", action=action)
        else:
            parser.add_argument(
                f"--{hp.name}", type=type(hp.default_value), default=hp.default_value
            )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/input_images",
        help="Path to input images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/synthetic_output",
        help="Directory to save generated data.",
    )

    args = parser.parse_args()

    # Create a configuration from the parsed arguments
    config_dict = {hp.name: getattr(args, hp.name) for hp in cs.get_hyperparameters()}
    config = Configuration(cs, values=config_dict)

    run_synthesis_pipeline(config, args.dataset_path, args.output_dir)


if __name__ == "__main__":
    main()
