import os
import json
import jax
import jax.numpy as jnp
from camar import camar_v0
from camar.maps import string_grid
from camar.render import svg_renderer
import numpy as np

# --- Configuration ---
OUTPUT_DIR = "output/camar_vqa_generation"
MAP_STR = """
.....#.....
.A.......G.
...........
.....#.....
..A.....G..
#.####.....
.....###.##
.G...#...A.
...........
.....#.....
..G..#..A..
"""
NUM_AGENTS = 4
SEED = 42


def generate_vqa_pairs(state, map_info):
    """
    Generates VQA pairs from the ground-truth environment state.
    """
    vqa_pairs = []
    agent_pos = np.array(state.physical_state.agent_pos)
    num_agents = agent_pos.shape[0]

    # --- VQA Pair 1: Agent Count ---
    vqa_pairs.append(
        {
            "id": "agent_count",
            "question": "How many agents are present in the scene?",
            "answer": str(num_agents),
        }
    )

    # --- VQA Pair 2: Closest Agent to a corner (e.g., top-left) ---
    top_left_corner = jnp.array([0.0, 0.0])
    distances_to_corner = jnp.linalg.norm(agent_pos - top_left_corner, axis=1)
    closest_agent_idx = int(jnp.argmin(distances_to_corner))
    vqa_pairs.append(
        {
            "id": f"closest_to_corner_0_0",
            "question": "Which agent is closest to the top-left corner?",
            "answer": f"agent {closest_agent_idx}",
        }
    )

    # --- VQA Pair 3: Distance between two agents ---
    if num_agents >= 2:
        agent_0_pos = agent_pos[0]
        agent_1_pos = agent_pos[1]
        distance = jnp.linalg.norm(agent_0_pos - agent_1_pos)
        vqa_pairs.append(
            {
                "id": "distance_agents_0_1",
                "question": "What is the distance between agent 0 and agent 1?",
                "answer": f"{distance:.2f} units",
            }
        )

    # --- VQA Pair 4: Relative position ---
    if num_agents >= 2:
        agent_0_pos = agent_pos[0]
        agent_1_pos = agent_pos[1]
        relative_pos = "left of" if agent_0_pos[0] < agent_1_pos[0] else "right of"
        vqa_pairs.append(
            {
                "id": "relative_pos_0_1",
                "question": "Is agent 0 to the left or right of agent 1?",
                "answer": relative_pos,
            }
        )

    return vqa_pairs


def main():
    """
    Main function to run the simulation and generate data.
    """
    print(f"Starting CAMAR VQA generation experiment...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Initialize CAMAR Environment
    key = jax.random.key(SEED)
    key, key_r = jax.random.split(key)

    # Use a string_grid map for a reproducible layout.
    # CAMAR places agents randomly within spawn areas ('A') for each reset.
    # We use a fixed seed for reproducibility of these random placements.
    custom_map = string_grid(map_str=MAP_STR, num_agents=NUM_AGENTS)
    env = camar_v0(custom_map)

    reset_fn = jax.jit(env.reset)

    # 2. Reset environment to get initial state
    print("Resetting environment...")
    _, state = reset_fn(key_r)

    # 3. Generate VQA pairs from the ground-truth state
    print("Generating VQA pairs...")
    # The map_info dictionary contains static info like goal positions
    vqa_data = generate_vqa_pairs(state, env.map.map_info)

    scene_info = {
        "scene_id": f"camar_scene_{SEED}",
        "vqa_pairs": vqa_data,
        "ground_truth_state": {
            "agent_pos": np.array(state.physical_state.agent_pos).tolist(),
            "goal_pos": np.array(env.map.map_info["goal_pos"]).tolist(),
        },
    }

    # 4. Save VQA data to JSON
    json_path = os.path.join(OUTPUT_DIR, "vqa_data.json")
    with open(json_path, "w") as f:
        json.dump(scene_info, f, indent=2)
    print(f"Saved VQA data to {json_path}")

    # 5. Render the scene to SVG
    print("Rendering scene to SVG...")
    renderer = svg_renderer(env)
    svg_content = renderer.render(state)
    svg_path = os.path.join(OUTPUT_DIR, "scene.svg")
    with open(svg_path, "w") as f:
        f.write(svg_content)
    print(f"Saved SVG rendering to {svg_path}")
    print("Experiment finished successfully.")


if __name__ == "__main__":
    main()
