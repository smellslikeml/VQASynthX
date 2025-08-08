# Maze Navigation Benchmark

This experiment evaluates a language model's ability to perform spatial reasoning and sequential decision-making in a text-based maze navigation task.

## Description

The `run.py` script simulates a maze environment where an agent must navigate from a starting point to a goal. The key challenge is that the agent has **limited visibility**: at any given position, it only knows how far it can travel in the four cardinal directions (North, East, South, West) before hitting a wall.

This setup is inspired by the [Maze Navigation Web Game](https://github.com/haffi112/maze-navigation-game), which was used to benchmark LLM spatial reasoning.

The current implementation includes a placeholder agent that follows a simple deterministic "left-hand rule" for demonstration and testing purposes. The `get_llm_move` function is designed to be replaced with a call to an actual language model API.

## How to Run

1.  **Build the Docker image:**

    ```sh
    docker build -t maze-eval-exp -f experiments/maze_eval/Dockerfile .
    ```

2.  **Run the experiment:**

    ```sh
    docker run --rm maze-eval-exp
    ```

The script will print the agent's progress at each step and a final summary of the outcome (success or failure), total moves, and the path taken.
