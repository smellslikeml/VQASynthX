import subprocess
import sys


def main():
    """
    Runs the POBAX PPO agent on the Navix-DMLab-Maze-01-v0 environment.

    This script serves as a simple entrypoint to benchmark a standard RL agent
    on a task requiring spatial reasoning and memory, integrating the POBAX
    benchmark suite into the experimental-vqasynth framework.

    The arguments passed to the PPO script are chosen for a quick, debug-level
    run to verify the integration.
    """
    # Command to run the PPO agent from the pobax library
    # Using Navix-DMLab-Maze-01-v0 as it's a good test of spatial uncertainty.
    # Arguments are set for a small-scale test run.
    command = [
        sys.executable,  # Use the current python interpreter
        "-m",
        "pobax.algos.ppo",
        "--env",
        "Navix-DMLab-Maze-01-v0",
        "--total_steps",
        "250000",
        "--num_envs",
        "16",
        "--learning_rate",
        "0.00025",
        "--debug",
        "--platform",
        "gpu",
    ]

    print(f"Running command: {' '.join(command)}")

    # Execute the command and stream output
    try:
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process:
            for line in process.stdout:
                print(line, end="")

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, process.args)

        print("\nPOBAX run completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\nPOBAX run failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("Error: python command not found. Is it in your PATH?")
        sys.exit(1)


if __name__ == "__main__":
    main()
