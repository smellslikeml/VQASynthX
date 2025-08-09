import json

# This is a mock LLM call for demonstration purposes.
# In a real implementation, this would be a call to an actual LLM API.
def mock_llm(prompt):
    """Mocks a Large Language Model call based on keywords in the prompt."""
    print("=" * 50)
    print("LLM PROMPT:")
    print(prompt)
    print("=" * 50)

    if "GENERATE-ACTIONS" in prompt:
        return json.dumps({
            "actions": [
                {"action_id": "A1", "thought": "First, I need to locate the red car to establish a reference point.", "executable_tool": "locate_object('red car')"},
                {"action_id": "A2", "thought": "Then, I should find the blue truck.", "executable_tool": "locate_object('blue truck')"},
                {"action_id": "A3", "thought": "Finally, I can determine the spatial relationship between them.", "executable_tool": "get_relation('red car', 'blue truck', 'in front of')"},
                {"action_id": "A4", "thought": "Alternatively, I could try to locate the truck first, but the car seems more prominent.", "executable_tool": "locate_object('blue truck')"},
            ]
        })
    elif "SIMULATE-OUTCOMES" in prompt:
        if "'A1'" in prompt:
            return json.dumps({"outcome": "I will have the coordinates and bounding box for the red car. This is a good first step.", "confidence": 0.9})
        if "'A2'" in prompt:
            return json.dumps({"outcome": "I will have the coordinates and bounding box for the blue truck. This is also a necessary step.", "confidence": 0.8})
        if "'A3'" in prompt:
            return json.dumps({"outcome": "I can only do this after finding both objects. This action is premature.", "confidence": 0.2})
        if "'A4'" in prompt:
             return json.dumps({"outcome": "Locating the truck first is possible, but the car is larger and easier to find, making A1 a better starting point.", "confidence": 0.7})
    elif "CHOOSE-BEST-ACTION" in prompt:
        return json.dumps({"best_action_id": "A1", "reasoning": "Action A1 has the highest confidence and establishes a primary reference. A3 is not yet possible. A1 is slightly better than A4 as a starting point."})
    elif "EXECUTE" in prompt:
        if "locate_object('red car')" in prompt:
            return json.dumps({"status": "SUCCESS", "result": "Red car located at bounding_box [100, 150, 300, 400]."})
    return "{}"


class WorldModelReasoner:
    """
    Applies simulation-based planning inspired by llm-reasoners's World Model
    to solve spatial reasoning VQA tasks.
    """
    def __init__(self, question):
        self.question = question
        self.max_steps = 5
        self.state = {
            "question": self.question,
            "known_info": [],
            "history": []
        }

    def plan_and_solve(self):
        """
        Runs the planning loop to generate a reasoning trace and find the answer.
        """
        for step in range(self.max_steps):
            print(f"\n--- STEP {step+1} ---")
            
            # 1. Propose potential next actions
            actions_prompt = f"""
            # INSTRUCTION: GENERATE-ACTIONS
            Current State: {json.dumps(self.state, indent=2)}
            Based on the current state, propose a few distinct actions to take next to answer the question.
            Provide your output as a JSON object with a key 'actions', which is a list of objects, each with 'action_id', 'thought', and 'executable_tool'.
            """
            action_proposals_str = mock_llm(actions_prompt)
            action_proposals = json.loads(action_proposals_str)

            # 2. Simulate outcomes for each proposed action (World Model)
            simulations = []
            for action in action_proposals["actions"]:
                single_sim_prompt = f"# INSTRUCTION: SIMULATE-OUTCOMES\nConsidering action '{action['action_id']}', what is the likely outcome?"
                simulation_str = mock_llm(single_sim_prompt)
                simulations.append(json.loads(simulation_str))

            # 3. Select the best action based on simulations (Policy)
            policy_prompt = f"""
            # INSTRUCTION: CHOOSE-BEST-ACTION
            Question: {self.question}
            Proposed Actions: {json.dumps(action_proposals['actions'], indent=2)}
            Simulated Outcomes: {json.dumps(simulations, indent=2)}
            Based on the simulations, which action is the best to take right now?
            Provide your output as a JSON object with 'best_action_id' and 'reasoning'.
            """
            choice_str = mock_llm(policy_prompt)
            choice = json.loads(choice_str)
            best_action_id = choice['best_action_id']
            best_action = next((a for a in action_proposals["actions"] if a["action_id"] == best_action_id), None)

            if not best_action:
                print("Could not determine best action. Stopping.")
                break

            print(f"\nSELECTED ACTION: {best_action['thought']}")

            # 4. Execute the chosen action
            execution_prompt = f"# INSTRUCTION: EXECUTE\nTool Call: {best_action['executable_tool']}"
            execution_result_str = mock_llm(execution_prompt)
            execution_result = json.loads(execution_result_str)

            # 5. Update state
            self.state["history"].append({
                "step": step + 1,
                "action": best_action,
                "result": execution_result
            })
            self.state["known_info"].append(execution_result['result'])

            # In a real scenario, we would check for a 'finish' action.
            if "get_relation" in best_action["executable_tool"]:
                print("\nFinal answer could now be determined.")
                break
        
        print("\n--- FINAL STATE ---")
        print(json.dumps(self.state, indent=2))
        return self.state


if __name__ == "__main__":
    sample_question = "Is the red car in front of the blue truck?"
    reasoner = WorldModelReasoner(question=sample_question)
    final_state = reasoner.plan_and_solve()
