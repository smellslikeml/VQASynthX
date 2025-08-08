import gymnasium as gym
import memory_gym
import json
import argparse
import os
from PIL import Image
import numpy as np

COMMAND_MAP = {
    0: "right",
    1: "down",
    2: "left",
    3: "up",
    4: "stay",
    5: "right down",
    6: "right up",
    7: "left down",
    8: "left up",
}

def generate_questions_for_episode(commands, image_dir_relative, episode_id):
    """Generates a list of question-answer pairs for a single episode."""
    qa_pairs = []
    command_strs = [COMMAND_MAP[c] for c in commands]

    # Use a placeholder image for questions about the full sequence, as no single frame represents it.
    # A more advanced generator could create a collage of the command frames.
    placeholder_img_path = os.path.join(image_dir_relative, "command_0_display.png")

    # Question 1: What is the full sequence?
    qa_pairs.append({
        "id": f"{episode_id}_q1",
        "image": placeholder_img_path,
        "conversations": [
            {"from": "human", "value": "What is the full sequence of commands the agent must remember?"},
            {"from": "gpt", "value": f"The agent must remember the following sequence: {', '.join(command_strs)}."}
        ]
    })

    # Question 2: How many commands?
    qa_pairs.append({
        "id": f"{episode_id}_q2",
        "image": placeholder_img_path,
        "conversations": [
            {"from": "human", "value": "How many commands are in the sequence?"},
            {"from": "gpt", "value": f"There are {len(command_strs)} commands in the sequence."}
        ]
    })

    # Question 3 to N: What is the Nth command?
    for i, cmd_str in enumerate(command_strs):
        ordinal = f"{i+1}{'st' if i+1==1 else 'nd' if i+1==2 else 'rd' if i+1==3 else 'th'}"
        qa_pairs.append({
            "id": f"{episode_id}_q{i+3}",
            "image": os.path.join(image_dir_relative, f"command_{i}_display.png"),
            "conversations": [
                {"from": "human", "value": f"What is the {ordinal} command in the sequence shown in the image?"},
                {"from": "gpt", "value": f"The {ordinal} command is to move {cmd_str}."}
            ]
        })

    return qa_pairs

def run_episode(env, output_base_dir, episode_idx):
    """Runs a single episode, saves frames, and returns the generated QA data."""
    episode_id = f"ep_{episode_idx}"
    image_dir_abs = os.path.join(output_base_dir, 'images', episode_id)
    os.makedirs(image_dir_abs, exist_ok=True)

    obs, info = env.reset()
    commands = info["commands"]
    done = False
    
    current_command_idx = 0
    
    # Run the episode to capture frames where commands are shown
    while not done:
        # The info dict tells us when the command tile is being displayed.
        # We save the frame on the first step of its appearance.
        if info.get("command_tile_timer", 0) == 1:
            if current_command_idx < len(commands):
                rgb_array = env.render()
                img = Image.fromarray(rgb_array)
                img_path = os.path.join(image_dir_abs, f"command_{current_command_idx}_display.png")
                img.save(img_path)
                current_command_idx += 1

        # Use a non-moving agent for simplicity; we only care about the command sequence.
        action = 4 # 'stay'
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # Generate questions based on the episode's command sequence
    image_dir_relative = os.path.join('images', episode_id)
    qa_data = generate_questions_for_episode(commands, image_dir_relative, episode_id)
    return qa_data

def main():
    parser = argparse.ArgumentParser(description="Generate VQA data from Mortar Mayhem.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output JSONL file.")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to generate.")
    parser.add_argument("--command_count", type=int, default=5, help="Number of commands per episode.")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use render_mode='rgb_array' to get frames.
    # We use the 'Endless' variant to align with the core concept of the source paper.
    env = gym.make(
        "Endless-MortarMayhem-v0",
        render_mode="rgb_array",
        options={"command_count": [args.command_count]}
    )

    with open(args.output_path, 'w') as f:
        for i in range(args.num_episodes):
            print(f"Generating episode {i+1}/{args.num_episodes}...")
            episode_qa_data = run_episode(env, output_dir, i)
            for item in episode_qa_data:
                f.write(json.dumps(item) + '\n')

    print(f"Successfully generated dataset.")
    print(f"Dataset saved to {args.output_path}")
    print(f"Images saved in {os.path.join(output_dir, 'images')}")

if __name__ == "__main__":
    main()
