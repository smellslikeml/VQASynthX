import random
import time
from typing import List, Dict, Any

# --- Core concepts from Symphony, simplified for a local PoC ---


class Task:
    """Represents a single unit of work, inspired by Symphony's TaskContract."""

    def __init__(
        self, description: str, required_capability: str, params: Dict[str, Any]
    ):
        self.description = description
        self.required_capability = required_capability
        self.params = params

    def __repr__(self):
        return f"Task(description='{self.description}', requires='{self.required_capability}')"


class Agent:
    """Represents a worker node with specific capabilities."""

    def __init__(self, node_id: str, capabilities: List[str]):
        self.node_id = node_id
        self.capabilities = capabilities

    def execute(self, task: Task) -> Dict[str, Any]:
        """A generic execution method that routes to the correct capability handler."""
        if task.required_capability not in self.capabilities:
            return {
                "status": "error",
                "message": f"Capability '{task.required_capability}' not supported by {self.node_id}",
            }

        handler_method = getattr(
            self, f"_handle_{task.required_capability.replace('-', '_')}", None
        )
        if handler_method:
            print(f"✔️  Agent '{self.node_id}' is executing task: {task.description}")
            return handler_method(task.params)
        else:
            return {
                "status": "error",
                "message": f"No handler found for '{task.required_capability}' on {self.node_id}",
            }

    def __repr__(self):
        return f"Agent(node_id='{self.node_id}', capabilities={self.capabilities})"


class CapabilityManager:
    """Simulates Symphony's decentralized ledger for agent capabilities."""

    def __init__(self):
        self.registry: Dict[str, List[Agent]] = {}

    def register(self, agent: Agent):
        """Registers an agent and its capabilities."""
        print(
            f"📢 Registering Agent '{agent.node_id}' with capabilities: {agent.capabilities}"
        )
        for cap in agent.capabilities:
            if cap not in self.registry:
                self.registry[cap] = []
            self.registry[cap].append(agent)

    def find_agents_for_capability(self, capability: str) -> List[Agent]:
        """Finds agents that can handle a specific capability."""
        return self.registry.get(capability, [])


# --- VQASynth-specific Agent implementations ---


class DepthEstimationAgent(Agent):
    """An agent specializing in depth estimation, a key stage in VQASynth."""

    def __init__(self, node_id: str):
        super().__init__(node_id, capabilities=["depth-estimation"])

    def _handle_depth_estimation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        image_path = params.get("image_path")
        print(f"    - Processing depth for '{image_path}'...")
        time.sleep(1)  # Simulate work
        return {
            "status": "success",
            "depth_map_path": f"/data/processed/{image_path.replace('.jpg', '_depth.png')}",
        }


class PromptGenerationAgent(Agent):
    """An agent specializing in generating VQA prompts, another VQASynth stage."""

    def __init__(self, node_id: str):
        super().__init__(node_id, capabilities=["prompt-generation", "text-generation"])

    def _handle_prompt_generation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        scene_data = params.get("scene_data")
        print(
            f"    - Generating prompts from scene data for image_id '{scene_data['image_id']}'..."
        )
        time.sleep(1.5)  # Simulate work
        return {
            "status": "success",
            "vqa_pairs": [{"q": "Is object A to the left of B?", "a": "Yes."}],
        }

    def _handle_text_generation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        prompt = params.get("prompt")
        print(
            f"    - Performing generic text generation for prompt: '{prompt[:30]}...'"
        )
        time.sleep(0.5)
        return {"status": "success", "text": "This is a generated response."}


# --- Symphony's Task Routing Logic ---


def route_and_execute_task(task: Task, manager: CapabilityManager) -> Dict[str, Any]:
    """Simulates Symphony's beacon-based routing to find and execute a task."""
    print(
        f"\n📡 Broadcasting beacon for task requiring '{task.required_capability}'..."
    )

    # 1. Capability Matching
    candidate_agents = manager.find_agents_for_capability(task.required_capability)

    if not candidate_agents:
        print(
            f"❌ No agents found with capability '{task.required_capability}'. Task failed."
        )
        return {"status": "failed", "message": "No suitable agent found"}

    # 2. Agent Selection (simplified: pick one at random)
    selected_agent = random.choice(candidate_agents)
    print(f"🎯 Matched task to Agent '{selected_agent.node_id}'")

    # 3. Task Execution
    result = selected_agent.execute(task)
    print(f"📊 Task completed by '{selected_agent.node_id}' with result: {result}")
    return result


# --- Main experiment execution ---

if __name__ == "__main__":
    print("--- Starting Symphony-inspired VQASynth Task Routing PoC ---")

    # 1. Initialize the capability manager (the "network ledger")
    capability_manager = CapabilityManager()

    # 2. Initialize specialized agents
    depth_agent = DepthEstimationAgent(node_id="depth_worker_01")
    prompt_agent = PromptGenerationAgent(node_id="prompt_worker_01")

    # 3. Register agents to the network
    capability_manager.register(depth_agent)
    capability_manager.register(prompt_agent)

    print("\n--- Network is live. Submitting tasks. ---")

    # 4. Create and submit a task that requires depth estimation
    depth_task = Task(
        description="Generate a depth map for a scene image",
        required_capability="depth-estimation",
        params={"image_path": "warehouse_sample_1.jpeg"},
    )
    route_and_execute_task(depth_task, capability_manager)

    # 5. Create and submit a task that requires prompt generation
    prompt_task = Task(
        description="Create VQA pairs from processed scene data",
        required_capability="prompt-generation",
        params={"scene_data": {"image_id": "warehouse_sample_1", "objects": 5}},
    )
    route_and_execute_task(prompt_task, capability_manager)

    # 6. Create and submit a task that only one agent can do (text-generation)
    text_task = Task(
        description="Generate a summary",
        required_capability="text-generation",
        params={"prompt": "Summarize the findings."},
    )
    route_and_execute_task(text_task, capability_manager)

    # 7. Create and submit a task that no agent can do
    unsupported_task = Task(
        description="Perform audio analysis",
        required_capability="audio-analysis",
        params={"audio_file": "sound.wav"},
    )
    route_and_execute_task(unsupported_task, capability_manager)

    print("\n--- Experiment finished ---")
