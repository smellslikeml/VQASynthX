import os
import random
import json
import logging
from openai import OpenAI
from typing import List, Dict, Any, Tuple

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prompt_evolution.log"),
        logging.StreamHandler()
    ]
)

# --- LLM Client Setup ---
try:
    client = OpenAI()
    LLM_MODEL = "gpt-4o-mini"
except Exception as e:
    logging.error(f"OpenAI client failed to initialize. Make sure OPENAI_API_KEY is set. Error: {e}")
    client = None

# --- Core Prompts (Inspired by MPaGE's methodology) ---

SYSTEM_PROMPT = """
You are an expert Python programmer specializing in generating code for AI data synthesis. Your task is to write and modify Python functions.

The functions you generate will have the following signature:
`def generate_vqa(scene_objects: List[Dict[str, Any]]) -> Tuple[str, str]:`

- `scene_objects` is a list of dictionaries, where each dictionary represents an object with keys like 'name', 'color', '3d_coords'.
- The function must return a tuple containing a (question, answer) pair.
- The function should contain logic to select objects and formulate a question about their spatial relationship.
- The function must be self-contained and not rely on external libraries beyond standard Python.
"""

MUTATION_PROMPT_TEMPLATE = """
Here is a Python function that generates a VQA pair. Your task is to mutate it to generate a new, logically different but still valid function. Introduce a new spatial concept (e.g., 'between', 'behind', 'furthest') or change the selection logic.

Original Function:
```python
{function_code}
```

Return only the complete, new Python function code. Do not add any explanations or markdown fences.
"""

CROSSOVER_PROMPT_TEMPLATE = """
Here are two Python functions that generate VQA pairs. Your task is to combine their logic to create a novel function. The new function should integrate ideas from both parents to ask a more complex or unique question.

Function A:
```python
{function_code_a}
```

Function B:
```python
{function_code_b}
```

Return only the complete, new Python function code that synthesizes logic from both. Do not add any explanations or markdown fences.
"""

EVALUATION_PROMPT_TEMPLATE = """
You are an expert in evaluating AI training data. Below is a VQA pair generated for a scene. Evaluate its quality based on the following criteria:
1.  **Clarity**: Is the question unambiguous?
2.  **Complexity**: Does it involve non-trivial spatial reasoning?
3.  **Relevance**: Is the question interesting for teaching an AI about spatial relationships?

Scene Objects:
{scene_objects}

Generated VQA:
- Question: "{question}"
- Answer: "{answer}"

Return a JSON object with a single key "score" from 1 (poor) to 10 (excellent).
"""

# --- Initial Population ---

INITIAL_POPULATION = [
"""
def generate_vqa(scene_objects: List[Dict[str, Any]]) -> Tuple[str, str]:
    if len(scene_objects) < 2:
        return ("Not enough objects for a spatial question.", "N/A")
    obj1 = random.choice(scene_objects)
    obj2 = random.choice([o for o in scene_objects if o != obj1])
    
    dist = ((obj1['3d_coords'][0] - obj2['3d_coords'][0])**2 + (obj1['3d_coords'][1] - obj2['3d_coords'][1])**2)**0.5
    
    question = f"How far is the {obj1['name']} from the {obj2['name']}?"
    answer = f"Approximately {dist:.2f} meters."
    return question, answer
""",
"""
def generate_vqa(scene_objects: List[Dict[str, Any]]) -> Tuple[str, str]:
    if not scene_objects:
        return ("The scene is empty.", "N/A")
    highest_obj = max(scene_objects, key=lambda o: o['3d_coords'][2])
    question = f"What is the highest object in the scene?"
    answer = f"The {highest_obj['name']}."
    return question, answer
"""
]

# --- Mock Scene Data ---
MOCK_SCENE_OBJECTS = [
    {'name': 'red ball', 'color': 'red', '3d_coords': [1.2, 3.4, 0.5]},
    {'name': 'blue box', 'color': 'blue', '3d_coords': [2.0, 1.0, 0.2]},
    {'name': 'green chair', 'color': 'green', '3d_coords': [-0.5, 2.5, 1.1]},
    {'name': 'floor', 'color': 'brown', '3d_coords': [0.0, 0.0, 0.0]}
]

# --- Evolutionary Algorithm Core ---

def execute_function(code_str: str, scene_objects: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Safely execute the generated Python code."""
    try:
        exec_globals = {'random': random}
        exec(code_str, exec_globals)
        func = exec_globals['generate_vqa']
        return func(scene_objects)
    except Exception as e:
        logging.warning(f"Function execution failed: {e}\nCode:\n{code_str}")
        return ("Execution error.", str(e))

def evaluate_population(population: List[str], scene_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Evaluate each individual in the population."""
    scored_population = []
    for i, code_str in enumerate(population):
        question, answer = execute_function(code_str, scene_objects)
        if question == "Execution error.":
            score = 0
        else:
            try:
                prompt = EVALUATION_PROMPT_TEMPLATE.format(
                    scene_objects=json.dumps(scene_objects, indent=2),
                    question=question,
                    answer=answer
                )
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant that provides JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                score = json.loads(response.choices[0].message.content).get('score', 0)
            except Exception as e:
                logging.error(f"LLM-based evaluation failed: {e}")
                score = 1 # Give a low score on failure
        
        logging.info(f"Individual {i} | Score: {score} | Question: {question}")
        scored_population.append({'code': code_str, 'score': score})
    return scored_population

def create_new_generation(scored_population: List[Dict[str, Any]], pop_size: int, mutation_rate: float) -> List[str]:
    """Create a new generation via selection, crossover, and mutation."""
    # Elitism: keep the best individuals
    scored_population.sort(key=lambda x: x['score'], reverse=True)
    num_elites = 2
    new_population = [ind['code'] for ind in scored_population[:num_elites]]

    # Generate the rest of the population
    while len(new_population) < pop_size:
        # Selection (Tournament)
        parent1 = random.choice(scored_population[:len(scored_population)//2])
        parent2 = random.choice(scored_population[:len(scored_population)//2])

        if random.random() < mutation_rate:
            # Mutation
            logging.info("Performing mutation...")
            prompt = MUTATION_PROMPT_TEMPLATE.format(function_code=parent1['code'])
            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL, messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                ])
                new_code = response.choices[0].message.content.strip('`python\n ')
                new_population.append(new_code)
            except Exception as e:
                logging.error(f"Mutation LLM call failed: {e}")

        else:
            # Crossover
            logging.info("Performing crossover...")
            prompt = CROSSOVER_PROMPT_TEMPLATE.format(function_code_a=parent1['code'], function_code_b=parent2['code'])
            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL, messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                ])
                new_code = response.choices[0].message.content.strip('`python\n ')
                new_population.append(new_code)
            except Exception as e:
                logging.error(f"Crossover LLM call failed: {e}")

    return new_population

def main():
    if not client:
        return

    # --- Hyperparameters ---
    POPULATION_SIZE = 10
    MAX_GENERATIONS = 5
    MUTATION_RATE = 0.4

    population = INITIAL_POPULATION
    # Ensure initial population is of the right size
    while len(population) < POPULATION_SIZE:
        population.append(random.choice(INITIAL_POPULATION))

    for gen in range(MAX_GENERATIONS):
        logging.info(f"\n--- Generation {gen + 1}/{MAX_GENERATIONS} ---")
        
        # 1. Evaluate current population
        scored_population = evaluate_population(population, MOCK_SCENE_OBJECTS)
        
        avg_score = sum(ind['score'] for ind in scored_population) / len(scored_population)
        logging.info(f"Generation {gen + 1} Average Score: {avg_score:.2f}")

        # 2. Create new generation
        population = create_new_generation(scored_population, POPULATION_SIZE, MUTATION_RATE)

    logging.info("\n--- Final Population ---")
    final_scored_population = evaluate_population(population, MOCK_SCENE_OBJECTS)
    for ind in sorted(final_scored_population, key=lambda x: x['score'], reverse=True):
        logging.info(f"Score: {ind['score']}\n{ind['code']}\n")

if __name__ == "__main__":
    main()
