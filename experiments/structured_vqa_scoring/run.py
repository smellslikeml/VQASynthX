import os
import json
import argparse
from openai import OpenAI

# Example VQA sample inspired by the VQASynth project's goals and output format.
VQA_SAMPLE = {
    "image_context": "An indoor warehouse scene. There is a red forklift on the left, and a stack of brown cardboard boxes on the right. A man in a red hat is walking near a wooden pallet.",
    "question": "How close is the man in red hat walking from the wooden pallet with boxes?",
    "answer": {
        "chain_of_thought": "1. Identify the 'man in red hat' and the 'wooden pallet with boxes' in the scene description. 2. Estimate their positions based on the context. The man is 'walking near' the pallet. 3. 'Near' implies a close proximity, likely within a few feet for safe movement in a warehouse. 4. Convert this colloquial understanding to a metric estimate. A reasonable estimate is about 3-5 feet. 5. Formulate the final answer based on this reasoning.",
        "final_answer": "The man in the red hat is walking approximately 4 feet away from the wooden pallet with boxes."
    }
}

EVALUATION_PROMPT_TEMPLATE = """
You are an expert AI Research Reviewer. Your task is to evaluate the quality of a synthetic Visual Question Answering (VQA) sample designed to teach spatial reasoning.

Evaluate the provided VQA sample based on the following 5 dimensions, which are adapted from a research proposal evaluation framework. For each dimension, provide a score from 1.0 to 10.0 and a brief justification.

**Evaluation Criteria:**

1.  **Relevance (1-10):** How well does the VQA pair test or teach a meaningful spatial reasoning concept (e.g., distance, orientation, relative position)?
2.  **Specificity (1-10):** How clear, unambiguous, and well-defined are the question and the final answer? Is there enough detail to be useful without being overly verbose?
3.  **Methodological Rigor (1-10):** How sound, logical, and correct is the 'chain_of_thought' reasoning? Does it correctly interpret the scene context to arrive at the answer?
4.  **Argumentative Cohesion (1-10):** How well does the final answer logically follow from the question and the reasoning process? Is the entire sample coherent?
5.  **Novelty (1-10):** How original or interesting is the spatial question? Does it go beyond simple object identification and probe a non-trivial spatial relationship?

**VQA Sample to Evaluate:**
```json
{vqa_sample}
```

**Instructions:**
Return a single JSON object with the following structure. Do not include any other text or explanations outside the JSON object.

{
  "Relevance": {{ "score": "X.X/10", "justification": "..." }},
  "Specificity": {{ "score": "X.X/10", "justification": "..." }},
  "Methodological_Rigor": {{ "score": "X.X/10", "justification": "..." }},
  "Argumentative_Cohesion": {{ "score": "X.X/10", "justification": "..." }},
  "Novelty": {{ "score": "X.X/10", "justification": "..." }},
  "Overall_Quality_Score": "X.X/10",
  "Decision": "Keep" or "Discard",
  "Justification_for_Decision": "A summary of why this sample should be kept or discarded based on the scores."
}
"""

def evaluate_vqa_quality(client: OpenAI, vqa_sample: dict, model: str = "gpt-4o"):
    """
    Uses an LLM to evaluate a VQA sample based on a structured rubric.
    """
    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        vqa_sample=json.dumps(vqa_sample, indent=2)
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert AI Research Reviewer. Your output must be a single, valid JSON object."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        evaluation = json.loads(response.choices[0].message.content)
        return evaluation
    except Exception as e:
        print(f"An error occurred during API call: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VQA sample quality using a structured, LLM-based rubric.")
    parser.add_argument("--openai-api-key", type=str, default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key for evaluation. Defaults to OPENAI_API_KEY environment variable.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="The model to use for evaluation.")
    args = parser.parse_args()

    if not args.openai_api_key:
        raise ValueError("OpenAI API key must be provided via --openai-api-key argument or OPENAI_API_KEY environment variable.")

    print("--- VQA Sample to be Evaluated ---")
    print(json.dumps(VQA_SAMPLE, indent=2))
    print("\n" + "="*40 + "\n")

    client = OpenAI(api_key=args.openai_api_key)

    print(f"Requesting evaluation from model: {args.model}...")
    evaluation_result = evaluate_vqa_quality(client, VQA_SAMPLE, model=args.model)

    if evaluation_result:
        print("--- Evaluation Result ---")
        print(json.dumps(evaluation_result, indent=2))
    else:
        print("Evaluation failed.")
