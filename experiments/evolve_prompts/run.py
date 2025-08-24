import os
import random
import json
import logging
from openai import OpenAI

# --- Configuration ---
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

CLIENT = OpenAI(api_key=API_KEY)
EVOLUTION_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o"  # Use a more powerful model for evaluation
LOG_FILE = "evolution.log"
OUTPUT_FILE = "evolved_prompts.json"

# --- Experiment Parameters ---
POPULATION_SIZE = 10
NUM_GENERATIONS = 5
ELITISM_COUNT = 2  # Number of top individuals to carry over directly
MUTATION_RATE = 0.5  # Probability of mutating an individual
CROSSOVER_RATE = 0.4  # Probability of crossing over two individuals

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)

# --- Prompts (inspired by MPaGE's use of LLM for code generation/modification) ---

# This is a stand-in for the scene data from the main VQASynth pipeline
DUMMY_SCENE_DATA = {
    "objects": [
        {"name": "red block", "id": "object1", "position": [0, 0, 1]},
        {"name": "blue sphere", "id": "object2", "position": [2, 0, 1]},
        {"name": "green pyramid", "id": "object3", "position": [-1, 3, 1]},
        {"name": "floor", "id": "surface1", "position": [0, 0, 0]},
    ]
}

INITIAL_POPULATION = [
    "How far is {object1} from {object2}?",
    "Is {object1} to the left of {object2}?",
    "What color is the object closest to {object1}?",
    "How many objects are on the {surface1}?",
    "Describe the position of {object1} relative to {object2}.",
]

# The "meta-prompt" to generate a sample question from a template
GENERATION_PROMPT = """
Given the following prompt template and scene data, generate a concrete and natural-sounding question.
Fill the placeholders like {{object1}} with specific object names from the scene data.

Template: "{template}"
Scene Data:
{scene_data}

Generated Question:
"""

# The "meta-prompt" for the Judge LLM to evaluate a generated question
# This replaces the complex evaluation functions in MPaGE's optimization tasks.
EVALUATION_PROMPT = """
You are an expert in designing training data for Vision Language Models.
Your task is to evaluate a generated question based on its potential to improve a model's spatial reasoning.
Please provide a score from 1 (poor) to 10 (excellent) for each of the following criteria, along with a brief justification.
Return the result as a JSON object with keys "spatial_relevance", "clarity", "difficulty", and "justification".

- Spatial Relevance: How well does the question probe spatial understanding (e.g., distance, orientation, relative position)?
- Clarity: Is the question unambiguous and easy to understand?
- Perceived Difficulty: How much reasoning is required to answer this question, assuming one has the 3D scene information?

Question to evaluate: "{question}"

JSON Evaluation:
"""

# The "meta-prompt" for the Mutation operator
MUTATION_PROMPT = """
You are an expert prompt engineer. Your task is to slightly mutate the following VQA prompt template to create a new, related variant.
The new template should still be a valid question but explore a slightly different aspect of spatial reasoning.
Make it more specific, more general, or change the relationship being queried.
Do not just rephrase; introduce a new logical element if possible.

Original Template:
"{template}"

Mutated Template:
"""

# The "meta-prompt" for the Crossover operator
CROSSOVER_PROMPT = """
You are an expert prompt engineer. Your task is to combine the conceptual elements of two VQA prompt templates into a single, new, coherent template.
The new template should integrate ideas from both parents to create a more complex or nuanced spatial question.

Template A: "{template1}"
Template B: "{template2}"

New Combined Template:
"""

# --- Core Evolutionary Functions (adapting MPaGE's concepts) ---


def call_llm(prompt, model, temperature=0.7):
    """Generic LLM call function."""
    try:
        response = CLIENT.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"LLM API call failed: {e}")
        return None


def evaluate_template(template: str) -> float:
    """
    Evaluates a single prompt template.
    1. Generate a concrete question from the template.
    2. Ask the Judge LLM to score the question.
    3. Return the average score.
    """
    logging.info(f"Evaluating template: '{template}'")

    # 1. Generate a concrete question
    obj1, obj2 = random.sample(
        [
            obj["name"]
            for obj in DUMMY_SCENE_DATA["objects"]
            if "surface" not in obj["id"]
        ],
        2,
    )
    surface = DUMMY_SCENE_DATA["objects"][3]["name"]
    filled_template = template.format(object1=obj1, object2=obj2, surface1=surface)

    gen_prompt = GENERATION_PROMPT.format(
        template=filled_template, scene_data=json.dumps(DUMMY_SCENE_DATA, indent=2)
    )
    question = call_llm(gen_prompt, EVOLUTION_MODEL, 0.2)
    if not question:
        return 0.0

    # 2. Ask Judge LLM to score
    eval_prompt = EVALUATION_PROMPT.format(question=question)
    evaluation_str = call_llm(eval_prompt, JUDGE_MODEL, 0.1)

    try:
        eval_json = json.loads(evaluation_str)
        score = (
            eval_json.get("spatial_relevance", 0)
            + eval_json.get("clarity", 0)
            + eval_json.get("difficulty", 0)
        ) / 3.0
        logging.info(f"  - Generated Question: {question}")
        logging.info(f"  - Scores: {eval_json}")
        logging.info(f"  - Average Score: {score:.2f}")
        return score
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        logging.error(
            f"Failed to parse evaluation response: {e}\nResponse: {evaluation_str}"
        )
        return 0.0


def mutate(template: str) -> str:
    """Uses an LLM to mutate a prompt template."""
    logging.info(f"Mutating: '{template}'")
    prompt = MUTATION_PROMPT.format(template=template)
    mutated = call_llm(prompt, EVOLUTION_MODEL, 0.8)
    mutated = mutated.strip().strip('"')
    logging.info(f"  - Result: '{mutated}'")
    return mutated if mutated else template


def crossover(template1: str, template2: str) -> str:
    """Uses an LLM to perform crossover between two templates."""
    logging.info(f"Crossover between '{template1}' and '{template2}'")
    prompt = CROSSOVER_PROMPT.format(template1=template1, template2=template2)
    child = call_llm(prompt, EVOLUTION_MODEL, 0.8)
    child = child.strip().strip('"')
    logging.info(f"  - Result: '{child}'")
    return child if child else template1


# --- Main Evolutionary Loop ---


def main():
    logging.info("--- Starting VQA Prompt Template Evolution ---")

    # Initialize population
    population = list(set(INITIAL_POPULATION))
    while len(population) < POPULATION_SIZE:
        population.append(random.choice(INITIAL_POPULATION))

    for generation in range(NUM_GENERATIONS):
        logging.info(f"\n--- Generation {generation + 1}/{NUM_GENERATIONS} ---")

        scores = {template: evaluate_template(template) for template in population}

        sorted_population = sorted(
            scores.items(), key=lambda item: item[1], reverse=True
        )

        logging.info("\n--- Generation Report ---")
        for i, (template, score) in enumerate(sorted_population):
            logging.info(f"{i+1}. Score: {score:.2f} | Template: '{template}'")

        new_population = []

        elites = [template for template, score in sorted_population[:ELITISM_COUNT]]
        new_population.extend(elites)
        logging.info(f"\nElites carried over: {elites}")

        while len(new_population) < POPULATION_SIZE:
            parent1 = random.choice(sorted_population[: POPULATION_SIZE // 2])[0]
            parent2 = random.choice(sorted_population[: POPULATION_SIZE // 2])[0]

            if random.random() < CROSSOVER_RATE and parent1 != parent2:
                child = crossover(parent1, parent2)
                if random.random() < MUTATION_RATE:
                    child = mutate(child)
                new_population.append(child)
            else:
                child = mutate(parent1)
                new_population.append(child)

        population = list(set(new_population))
        while len(population) < POPULATION_SIZE:
            population.append(mutate(random.choice(elites)))

    logging.info("\n--- Final Population Evaluation ---")
    final_scores = {template: evaluate_template(template) for template in population}
    final_sorted_population = sorted(
        final_scores.items(), key=lambda item: item[1], reverse=True
    )

    logging.info("\n--- Final Evolved Templates ---")
    for i, (template, score) in enumerate(final_sorted_population):
        logging.info(f"{i+1}. Score: {score:.2f} | Template: '{template}'")

    output_data = [{"template": t, "score": s} for t, s in final_sorted_population]
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output_data, f, indent=2)
    logging.info(f"Saved best templates to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
