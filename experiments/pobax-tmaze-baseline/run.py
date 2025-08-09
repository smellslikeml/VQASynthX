import subprocess
import sys

def main():
    """
    Runs the recurrent PPO agent from the POBAX library on the T-Maze environment.
    This script is based on the usage examples provided in the POBAX README.
    It serves as a baseline for reinforcement learning experiments involving
    partial observability and memory.
    """
    # Command to run the PPO agent on T-Maze, as suggested by the POBAX docs.
    # We use a limited number of steps for a quick, testable experiment.
    # The --debug flag provides verbose logging.
    # The --platform gpu flag is recommended for CUDA-enabled environments.
    command = [
        sys.executable,
        "-m", "pobax.algos.ppo",
        "--env", "tmaze_5",
        "--total_steps", "250000",
        "--debug",
        "--platform", "gpu"
    ]

    print(f"Executing experiment command: {' '.join(command)}")

    try:
        # We use subprocess.run to execute the command.
        # `check=True` will raise a CalledProcessError if the command returns a non-zero exit code.
        # The output is streamed to the console in real-time.
        process = subprocess.run(
            command,
            check=True,
            text=True,
        )
        print("Experiment completed successfully.")

    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("Error: Could not execute 'python -m pobax.algos.ppo'.", file=sys.stderr)
        print("Please ensure the 'pobax' package is installed in the environment.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
