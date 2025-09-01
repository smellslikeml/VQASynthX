import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import logging

# --- Configuration ---
load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Constants & Prompts (Inspired by HAMLET) ---

# This description serves as the source text, analogous to a text chunk in HAMLET.
IMAGE_DESCRIPTION = """
In a brightly lit warehouse, a man wearing a red hard hat, a blue shirt, and jeans is walking near a stack of brown cardboard boxes on a wooden pallet. To the left, a large red forklift is parked with its forks lowered. The floor is a smooth, grey concrete, and in the background, there are metal shelves stacked high with more boxes and supplies, indicating a busy industrial environment. The lighting comes from large overhead fixtures, casting even light across the scene. The man appears to be inspecting the area.
"""

# This prompt adapts HAMLET's tree extraction concept for an image description.
# It asks for a hierarchical breakdown of facts about the scene.
TREE_EXTRACTION_PROMPT_TEMPLATE = """
You are an expert scene analyst. Your task is to extract a hierarchy of key facts from the provided image description. Structure the output as a JSON object containing a list of "roots".

Each "root" represents a major entity or concept in the scene.
Each "root" has "branches", which are more specific facts about the root.
Each "branch" has "leaves", which are the most granular details.

Each fact (root, branch, or leaf) should be a concise descriptive statement.

Here is the image description:
---
{description}
---

Produce a JSON object with a single key "roots" containing the list of key fact trees.
Example structure:
{
  "roots": [
    {
      "root_fact": "A man is present in the warehouse.",
      "branches": [
        {
          "branch_fact": "The man has specific attire.",
          "leaves": [
            {"leaf_fact": "He wears a red hard hat."},
            {"leaf_fact": "He wears a blue shirt and jeans."}
          ]
        },
        {
          "branch_fact": "The man is performing an action.",
          "leaves": [
            {"leaf_fact": "He is walking near stacked boxes."},
            {"leaf_fact": "He appears to be inspecting the area."}
          ]
        }
      ]
    }
  ]
}
"""

# This prompt adapts HAMLET's query generation concept.
# It uses the structured tree to generate questions at different granularities.
QUERY_GENERATION_PROMPT_TEMPLATE = """
You are an expert question generator for a Visual Question Answering (VQA) dataset.
Based on the provided hierarchical key-fact tree about a scene, generate one question for each fact (root, branch, and leaf).

- Root questions should be high-level.
- Branch questions should be more specific.
- Leaf questions should focus on fine-grained details.

The questions should be answerable from the original image description. Frame them as if asking someone looking at the image.
For each question, provide a concise answer based *only* on the fact statement.

Here is the key-fact tree JSON:
---
{tree}
---

Produce a JSON object containing a list of "qaps" (question-answer pairs). Each item in the list should correspond to a fact in the tree and include the `level` (root, branch, leaf), the `fact` statement, the generated `question`, and the `answer`.

Example Output Structure:
{
  "qaps": [
    {
      "level": "root",
      "fact": "A man is present in the warehouse.",
      "question": "Is there a person visible in the scene?",
      "answer": "Yes, a man is in the warehouse."
    },
    {
      "level": "branch",
      "fact": "The man has specific attire.",
      "question": "What is the man wearing?",
      "answer": "He is wearing a red hard hat, a blue shirt, and jeans."
    }
  ]
}
"""

# --- Core Logic ---


def get_llm_response(prompt, client):
    """Calls the OpenAI API to get a JSON response."""
    logging.info("Sending prompt to LLM...")
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        logging.info("Successfully received LLM response.")
        return json.loads(content)
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        return None


def extract_key_fact_tree(description, client):
    """Generates the hierarchical key-fact tree from a description."""
    logging.info("Step 1: Extracting key-fact tree.")
    prompt = TREE_EXTRACTION_PROMPT_TEMPLATE.format(description=description)
    tree_data = get_llm_response(prompt, client)
    return tree_data


def generate_queries_from_tree(tree_data, client):
    """Generates questions from the key-fact tree."""
    logging.info("Step 2: Generating queries from the tree.")
    prompt = QUERY_GENERATION_PROMPT_TEMPLATE.format(
        tree=json.dumps(tree_data, indent=2)
    )
    queries = get_llm_response(prompt, client)
    return queries


def main():
    """Main execution function."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error(
            "OPENAI_API_KEY environment variable not set. Please create a .env file or export it."
        )
        return

    client = OpenAI(api_key=api_key)

    # Step 1: Extract the tree from the image description
    key_fact_tree = extract_key_fact_tree(IMAGE_DESCRIPTION, client)
    if not key_fact_tree:
        logging.error("Failed to extract key-fact tree. Aborting.")
        return

    # Step 2: Generate queries from the extracted tree
    generated_qaps = generate_queries_from_tree(key_fact_tree, client)
    if not generated_qaps:
        logging.error("Failed to generate queries. Aborting.")
        return

    # Step 3: Combine results and save to a file
    final_output = {
        "source_description": IMAGE_DESCRIPTION,
        "key_fact_tree": key_fact_tree,
        "generated_qaps": generated_qaps,
    }

    output_path = "hamlet_vqa_synthesis_output.json"
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)

    logging.info(f"Successfully generated VQA data. Output saved to {output_path}")
    if generated_qaps.get("qaps"):
        logging.info(
            "Sample Question: " + generated_qaps["qaps"][0].get("question", "N/A")
        )


if __name__ == "__main__":
    main()
