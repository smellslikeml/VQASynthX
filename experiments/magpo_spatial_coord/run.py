import numpy as np

# A simplified representation of the "Guiding Policy" from MAGPO.
# It performs centralized assignment of tasks to agents.
def centralized_target_assigner(agent_positions, target_positions):
    """Assigns targets to agents to minimize total travel distance."""
    # In a 2-agent case, there are two assignment permutations.
    # Permutation 1: Agent 0 -> Target 0, Agent 1 -> Target 1
    dist1 = np.linalg.norm(agent_positions[0] - target_positions[0]) + \
            np.linalg.norm(agent_positions[1] - target_positions[1])

    # Permutation 2: Agent 0 -> Target 1, Agent 1 -> Target 0
    dist2 = np.linalg.norm(agent_positions[0] - target_positions[1]) + \
            np.linalg.norm(agent_positions[1] - target_positions[0])

    if dist1 <= dist2:
        print("Centralized Assigner: Agent 0 -> Target 0, Agent 1 -> Target 1")
        return {0: 0, 1: 1}
    else:
        print("Centralized Assigner: Agent 0 -> Target 1, Agent 1 -> Target 0")
        return {0: 1, 1: 0}

class SimpleAgent:
    """A simple agent that moves towards a target."""
    def __init__(self, agent_id, pos):
        self.id = agent_id
        self.pos = np.array(pos, dtype=float)
        self.target_pos = None

    def set_target(self, target_pos):
        self.target_pos = np.array(target_pos, dtype=float)

    def policy(self, observation):
        """Decentralized policy: move towards the assigned target."""
        if self.target_pos is None:
            return np.array([0, 0]) # Do nothing if no target

        direction = self.target_pos - self.pos
        # Normalize direction and move one step
        norm = np.linalg.norm(direction)
        if norm < 1.0: # If close enough, snap to target
            return direction
        return direction / norm

class MultiAgentEnv:
    """A simple multi-agent environment for spatial coordination."""
    def __init__(self, num_agents=2, world_size=10):
        self.num_agents = num_agents
        self.world_size = world_size
        # Place targets
        self.target_positions = [
            np.array([2.0, 2.0]),
            np.array([8.0, 8.0])
        ]
        # Place agents
        self.agents = [
            SimpleAgent(0, pos=[8.0, 2.0]),
            SimpleAgent(1, pos=[2.0, 8.0])
        ]
        print("Environment initialized.")
        print(f"Target 0 at {self.target_positions[0]}, Target 1 at {self.target_positions[1]}")
        print(f"Agent 0 starts at {self.agents[0].pos}, Agent 1 starts at {self.agents[1].pos}")

    def step(self):
        """Run one step of the simulation."""
        actions = []
        for agent in self.agents:
            # Observation for each agent is its own position (simplified)
            obs = agent.pos
            actions.append(agent.policy(obs))

        # Apply actions
        for i, agent in enumerate(self.agents):
            agent.pos += actions[i]

        # Check for success
        done = True
        for agent in self.agents:
            if np.linalg.norm(agent.pos - agent.target_pos) > 0.1:
                done = False
                break
        return done

def run_simulation():
    """Main simulation loop."""
    print("--- Starting MAGPO-inspired Spatial Coordination Experiment ---")
    max_steps = 20
    env = MultiAgentEnv()

    # Centralized assignment step (inspired by MAGPO's guided policy)
    agent_positions = [agent.pos for agent in env.agents]
    assignments = centralized_target_assigner(agent_positions, env.target_positions)
    for agent_id, target_id in assignments.items():
        env.agents[agent_id].set_target(env.target_positions[target_id])
        print(f"  -> Agent {agent_id} assigned target {target_id} at {env.target_positions[target_id]}")


    # Decentralized execution loop
    print("\n--- Starting Decentralized Execution ---")
    success = False
    for step in range(max_steps):
        done = env.step()
        print(f"Step {step+1}: Agent 0 at {env.agents[0].pos.round(2)}, Agent 1 at {env.agents[1].pos.round(2)}")
        if done:
            print(f"\nSuccess! All agents reached their targets in {step+1} steps.")
            success = True
            break

    if not success:
        print(f"\nFailure. Agents did not reach targets within {max_steps} steps.")

if __name__ == "__main__":
    run_simulation()
