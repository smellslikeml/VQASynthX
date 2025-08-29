import argparse
import importlib
import os
import sys

# Add the current directory to the path to allow relative imports within the container
sys.path.append(os.path.dirname(os.path.realpath(__file__)))


class Exp_Main:
    def __init__(self, args):
        self.args = args
        self.model = self._build_model()

    def _build_model(self):
        """Dynamically imports and instantiates the specified model."""
        try:
            # The package is 'models' relative to this script's location
            model_module = importlib.import_module(
                f".models.{self.args.model.lower()}", package="models"
            )
            Model = getattr(model_module, "Model")
        except (ImportError, AttributeError) as e:
            print(f"Error: Model '{self.args.model}' not found or module is invalid.")
            print(
                f"Please ensure 'docker/depth_stage/models/{self.args.model.lower()}.py' exists and contains a 'Model' class."
            )
            raise e

        model = Model(self.args)
        return model

    def process(self):
        """Executes the processing logic for the selected model."""
        print(f"Using model: {self.args.model}")
        print(f"Processing data from {self.args.input_dir} to {self.args.output_dir}")
        self.model.process(self.args.input_dir, self.args.output_dir)
        print("Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQASynth Depth Estimation Stage")

    # --- Basic Config (Inspired by TSLib's run.py) ---
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="VGGT",
        help="model name, e.g., VGGT, DepthPro",
    )

    # --- Data Config ---
    parser.add_argument(
        "--input_dir", type=str, required=True, help="path to input data"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="path to save output data"
    )

    # --- Model-specific args (example) ---
    parser.add_argument(
        "--vggt_specific_param",
        type=int,
        default=1,
        help="example model-specific parameter for VGGT",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    exp = Exp_Main(args)
    exp.process()
