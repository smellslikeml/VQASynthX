import json

# Maze data structure inspired by maze_data.js from the source repository.
# A 5x5 maze for a minimal, self-contained test.
MAZE_5x5_0 = {
    "width": 5,
    "height": 5,
    "start": [0, 4],
    "goal": [4, 0],
    "cells": [
        [{'walls': {'north': True, 'east': False, 'south': False, 'west': True}},
         {'walls': {'north': True, 'east': True, 'south': True, 'west': False}},
         {'walls': {'north': True, 'east': False, 'south': False, 'west': True}},
         {'walls': {'north': True, 'east': False, 'south': False, 'west': False}},
         {'walls': {'north': True, 'east': True, 'south': False, 'west': False}}],
        [{'walls': {'north': False, 'east': True, 'south': False, 'west': True}},
         {'walls': {'north': True, 'east': False, 'south': False, 'west': True}},
         {'walls': {'north': False, 'east': True, 'south': False, 'west': False}},
         {'walls': {'north': False, 'east': True, 'south': True, 'west': True}},
         {'walls': {'north': False, 'east': True, 'south': False, 'west': True}}],
        [{'walls': {'north': False, 'east': False, 'south': True, 'west': True}},
         {'walls': {'north': False, 'east': True, 'south': True, 'west': False}},
         {'walls': {'north': False, 'east': False, 'south': False, 'west': True}},
         {'walls': {'north': True, 'east': False, 'south': False, 'west': False}},
         {'walls': {'north': False, 'east': True, 'south': False, 'west': False}}],
        [{'walls': {'north': True, 'east': False, 'south': False, 'west': True}},
         {'walls': {'north': True, 'east': False, 'south': True, 'west': False}},
         {'walls': {'north': False, 'east': False, 'south': True, 'west': False}},
         {'walls': {'north': False, 'east': False, 'south': False, 'west': False}},
         {'walls': {'north': False, 'east': True, 'south': True, 'west': False}}],
        [{'walls': {'north': False, 'east': False, 'south': True, 'west': True}},
         {'walls': {'north': True, 'east': False, 'south': True, 'west': False}},
         {'walls': {'north': True, 'east': False, 'south': True, 'west': False}},
         {'walls': {'north': False, 'east': False, 'south': True, 'west': False}},
         {'walls': {'north': True, 'east': True, 'south': True, 'west': False}}]
    ]
}

class MazeRunner:
    """Manages the state and logic of navigating a maze."""

    def __init__(self, maze_data, max_moves=100):
        self.maze = maze_data
        self.width = maze_data['width']
        self.height = maze_data['height']
        self.start_pos = tuple(maze_data['start'])
        self.goal_pos = tuple(maze_data['goal'])

        self.player_pos = self.start_pos
        self.moves = 0
        self.max_moves = max_moves
        self.path = [self.player_pos]
        self.visited = {self.player_pos: 1}
        self.facing = 'NORTH' # For relative turn logic in placeholder agent

    def get_visibility(self):
        """Replicates the limited visibility logic from source game.js.
        
        For each direction from the current cell, count how many steps
        are possible until a wall is hit.
        """
        visibility = {}
        directions = {
            'NORTH': (0, -1),
            'EAST': (1, 0),
            'SOUTH': (0, 1),
            'WEST': (-1, 0)
        }

        for name, (dx, dy) in directions.items():
            count = 0
            x, y = self.player_pos
            while True:
                cell = self.maze['cells'][y][x]
                if cell['walls'][name.lower()]:
                    break
                
                x += dx
                y += dy

                if not (0 <= x < self.width and 0 <= y < self.height):
                    break
                
                count += 1
            visibility[name] = count
            
        return visibility

    def format_prompt(self, visibility):
        """Creates the textual prompt for the LLM based on current state."""
        prompt = (
            f"You are an agent in a {self.width}x{self.height} maze. Your goal is to reach {self.goal_pos}.\n"
            f"You are currently at position {self.player_pos}. You have made {self.moves} moves.\n"
            f"Your visit counts are: {self.visited}.\n\n"
            f"From your current position, you can see:\n"
            f"- NORTH: {visibility['NORTH']} steps clear.\n"
            f"- EAST: {visibility['EAST']} steps clear.\n"
            f"- SOUTH: {visibility['SOUTH']} steps clear.\n"
            f"- WEST: {visibility['WEST']} steps clear.\n\n"
            f"Available moves are: {[d for d, v in visibility.items() if v > 0]}\n"
            f"Which direction do you move?"
        )
        return prompt

    def move(self, direction):
        """Updates the player's position based on the chosen direction."""
        directions = {'NORTH': (0, -1), 'EAST': (1, 0), 'SOUTH': (0, 1), 'WEST': (-1, 0)}
        if direction not in directions:
            print(f"Invalid direction: {direction}")
            return False

        dx, dy = directions[direction]
        px, py = self.player_pos
        cell = self.maze['cells'][py][px]

        if not cell['walls'][direction.lower()]:
            self.player_pos = (px + dx, py + dy)
            self.moves += 1
            self.path.append(self.player_pos)
            self.visited[self.player_pos] = self.visited.get(self.player_pos, 0) + 1
            self.facing = direction
            return True
        else:
            print(f"Cannot move {direction}, there is a wall.")
            return False

    def is_complete(self):
        return self.player_pos == self.goal_pos

    def is_over(self):
        return self.moves >= self.max_moves or self.is_complete()

def get_llm_move(runner, prompt):
    """Placeholder for an actual LLM call.
    
    This deterministic agent uses a "left-hand rule" relative to its last move.
    It tries to turn left, then go straight, then turn right, then turn back.
    This provides a simple, testable baseline.
    """
    print("--- AGENT PROMPT ---")
    print(prompt)
    print("--- END PROMPT ---")
    
    visibility = runner.get_visibility()
    valid_moves = [d for d, v in visibility.items() if v > 0]

    # Relative directions based on current facing direction
    turn_order = {
        'NORTH': ['WEST', 'NORTH', 'EAST', 'SOUTH'],
        'EAST':  ['NORTH', 'EAST', 'SOUTH', 'WEST'],
        'SOUTH': ['EAST', 'SOUTH', 'WEST', 'NORTH'],
        'WEST':  ['SOUTH', 'WEST', 'NORTH', 'EAST'],
    }

    for move in turn_order[runner.facing]:
        if move in valid_moves:
            return move
    
    # Should not happen in a valid maze, but as a fallback
    return valid_moves[0] if valid_moves else None

def main():
    print("Starting Maze Navigation Benchmark...")
    runner = MazeRunner(MAZE_5x5_0, max_moves=50)

    while not runner.is_over():
        visibility = runner.get_visibility()
        prompt = runner.format_prompt(visibility)
        
        # In a real scenario, this would be an API call to an LLM
        chosen_move = get_llm_move(runner, prompt)

        if not chosen_move:
            print("Agent is stuck. No moves available.")
            break

        print(f"Agent chose to move: {chosen_move}")
        runner.move(chosen_move)
        print(f"Step {runner.moves}: New position is {runner.player_pos}\n")

    print("--- BENCHMARK COMPLETE ---")
    if runner.is_complete():
        print(f"Success! Goal reached at {runner.goal_pos}.")
    else:
        print(f"Failure. Agent did not reach goal {runner.goal_pos}. Stopped at {runner.player_pos}.")

    print(f"Total Moves: {runner.moves}")
    print(f"Final Path: {runner.path}")
    print(f"Cell Visits: {json.dumps(runner.visited, default=str)}")

if __name__ == "__main__":
    main()
