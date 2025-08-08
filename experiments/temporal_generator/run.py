import gymnasium as gym
import memory_gym
import json
from pathlib import Path
import imageio
import numpy as np

# Command mapping based on memory_gym/mortar_mayhem.py logic and README
# 9 commands: right, down, left, up, stay, right down, right up, left down, left up
COMMAND_MAP = {
    0: "move right",
    1: "move down",
    2: "move left",
    3: "move up",
    4: "stay",
    5: "move down-right",
    6: "move up-right",
    7: "move down-left",
    8: "move up-left",
}

def generate_temporal_qa_sample(output_dir: Path, sample_id: int):
    """
    Generates a single temporal VQA sample from the MortarMayhem environment.
    A sample consists of a GIF showing the command sequence and a JSON
    file with the corresponding question and answer.
    """
    env = None
    try:
        # Using the non-endless version to have a defined command sequence phase
        # Override default params for predictable generation
        config = {
            "command_count": [4],
            "command_show_duration": [8],
            "command_show_delay": [4],
            "allowed_commands": 5 # Use simpler up, down, left, right, stay commands
        }
        env = gym.make(
            "MortarMayhem-v0",
            render_mode="rgb_array",
            config=config
        )
        
        obs, info = env.reset(seed=sample_id)
        
        # Ground truth information
        commands = info["commands"]
        command_texts = [COMMAND_MAP[c] for c in commands]
        
        # The VQA pair
        question = "Based on the visual cues presented in the video, what is the correct sequence of actions to perform?"
        answer = ", ".join(command_texts).capitalize() + "."
        
        frames = []
        
        # The total duration of the command showing phase
        show_phase_steps = (env.unwrapped.command_show_duration + env.unwrapped.command_show_delay) * env.unwrapped.command_count
        
        # Collect frames only during the command showing phase
        for i in range(show_phase_steps):
            frame = env.render()
            frames.append(frame)
            # Take a no-op action as we are just observing
            obs, reward, done, truncation, info = env.step(4) # Action 4 is 'stay'
            if done or truncation:
                break
        
        if not frames:
            print(f"Warning: No frames generated for sample {sample_id}. Skipping.")
            return

        # Create output directory for the sample
        sample_dir = output_dir / f"sample_{sample_id:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Save GIF
        gif_path = sample_dir / "sequence.gif"
        imageio.mimsave(gif_path, frames, fps=10)
        
        # Save QA JSON in a LLaVA-compatible format
        qa_data = {
            "id": f"temporal_mortar_mayhem_{sample_id:04d}",
            "image": str(gif_path.relative_to(output_dir)),
            "conversations": [
                {"from": "human", "value": f"<image>\n{question}"},
                {"from": "gpt", "value": answer},
            ]
        }
        
        json_path = sample_dir / "qa.json"
        with open(json_path, 'w') as f:
            json.dump(qa_data, f, indent=2)

        print(f"Successfully generated sample {sample_id}")

    finally:
        if env:
            env.close()


def main(num_samples=20, output_dir="output/temporal_qa_dataset"):
    """
    Main function to generate a dataset of temporal VQA samples.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_samples} temporal VQA samples in '{output_path}'...")
    
    for i in range(num_samples):
        generate_temporal_qa_sample(output_path, i)
        
    print(f"\nGeneration complete. Dataset created at ./{output_dir}")


if __name__ == "__main__":
    # This script will generate a small dataset of temporal VQA samples.
    # Each sample will be a directory containing a GIF of the Mortar Mayhem
    # instruction phase and a JSON file with a question about the sequence
    # and the ground-truth answer.
    main()
