import json
import random

# Mock LLM client for reproducibility. In a real scenario, this would be an OpenAI/Anthropic/etc. client.
class MockLLMClient:
    def generate(self, prompt):
        print("--- LLM PROMPT ---")
        print(prompt)
        print("--------------------")
        
        # This is a mocked response that simulates the LLM's output.
        # It combines elements from the elite prompts provided in the prompt.
        evolved_templates = [
            "What is the spatial relationship between {object_1} and the {object_2} that is {relation} the {object_3}?",
            "Considering the {surface} as a reference, is the {object_1} located further away than the {object_2}?",
            "Estimate the distance in {unit} between the {object_1} and {object_2}, and describe if {object_1} is to the {relation} of {object_2}.",
            "Which object is closer to the {object_1}: the {object_2} or the {object_3}?"
        ]
        return json.dumps(evolved_templates)

# --- MPaGE-inspired Evaluation ---
# We define objectives to score our prompt templates.
# A real implementation would be more sophisticated.

def evaluate_template(template: str):
    """
    Evaluates a prompt template based on complexity and diversity proxies.
    
    - Complexity: Number of placeholder variables (e.g., {object_1}).
    - Diversity: Number of unique types of placeholders (e.g., 'object', 'relation').
    """
    complexity_score = template.count('{')
    
    placeholders = [p.split('}')[0] for p in template.split('{')[1:]]
    unique_placeholder_types = set(p.split('_')[0] for p in placeholders)
    diversity_score = len(unique_placeholder_types)
    
    # Simple weighted score for demonstration. In MPaGE, this would be a multi-objective vector.
    return {"complexity": complexity_score, "diversity": diversity_score, "score": complexity_score + diversity_score}

# --- MPaGE-inspired Evolution Step ---

def evolve_prompts(elite_templates, llm_client):
    """
    Uses an LLM to perform "cross-cluster recombination" on elite templates,
    inspired by MPaGE's method of generating new heuristics.
    """
    prompt = f"""
You are an expert in generating high-quality Visual Question Answering (VQA) prompts for spatial reasoning.
Your task is to create new, more complex, and diverse question templates by analyzing and recombining the best-performing examples.

Here are the elite templates that have been selected for their high scores in complexity and diversity:
{json.dumps(elite_templates, indent=2)}

Please generate 4 new templates that:
1. Combine ideas and structures from the provided elite templates.
2. Introduce more complex spatial relationships or reasoning steps.
3. Maintain the placeholder format (e.g., {{object_1}}, {{relation}}).
4. Output ONLY a valid JSON list of strings.

Example of a good combination: Combining a "distance" query with a "relative position" query.

New Evolved Templates:
"""
    response = llm_client.generate(prompt)
    return json.loads(response)

def main():
    """
    Runs a single-generation experiment of evolving VQA prompt templates.
    """
    # 1. Initial Population (like in an evolutionary algorithm)
    initial_population = [
        "Is the {object_1} to the {relation} of the {object_2}?",
        "How far is the {object_1} from the {object_2}?",
        "What color is the {object_1}?", # A low-quality, non-spatial prompt
        "Is there a {object_1} in the image?",
        "Estimate the distance between {object_1} and {object_2} in {unit}.",
        "Using the {surface} as a reference, is the {object_1} above the {object_2}?"
    ]

    print("--- Initial Population Evaluation ---")
    evaluated_population = []
    for template in initial_population:
        scores = evaluate_template(template)
        evaluated_population.append({"template": template, "scores": scores})
        print(f'Template: "{template}" -> Scores: {scores}')

    # 2. Selection (like Pareto Front selection)
    # We select the top N individuals based on our combined score.
    evaluated_population.sort(key=lambda x: x['scores']['score'], reverse=True)
    
    selection_num = 3 # In MPaGE, this is `selection_num`
    elites = [p['template'] for p in evaluated_population[:selection_num]]
    
    print("\n--- Selected Elite Templates for Evolution ---")
    for elite in elites:
        print(f'- "{elite}"')

    # 3. LLM-driven Recombination/Mutation (The core of the MPaGE idea)
    print("\n--- Evolving New Templates with LLM ---")
    llm_client = MockLLMClient()
    new_templates = evolve_prompts(elites, llm_client)

    print("\n--- Newly Evolved Templates ---")
    for template in new_templates:
        scores = evaluate_template(template)
        print(f'Template: "{template}" -> Scores: {scores}')
        
    print("\nExperiment complete. The evolved templates show higher complexity and diversity, "
          "demonstrating the potential of the MPaGE-inspired approach for VQA prompt generation.")


if __name__ == "__main__":
    main()
