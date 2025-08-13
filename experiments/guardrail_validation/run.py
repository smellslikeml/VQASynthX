import os
import json
from openai import OpenAI
import backoff

# --- Configuration ---
# To run this, you must have an OPENAI_API_KEY environment variable set.
# The user will need to install the openai library: pip install openai backoff
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

client = OpenAI(api_key=API_KEY)

# --- Sample Data ---
# This data simulates the output of a previous pipeline stage (e.g., prompt_stage).
# It includes a mix of good and bad examples.
# The scene_context would be derived from the VQASynth pipeline (depth, objects, etc.).
SAMPLE_VQA_PAIRS = [
    {
        "id": "good_1",
        "image_id": "warehouse_sample_1.jpeg",
        "scene_context": "An image of a warehouse. A red forklift is on the left. A stack of brown cardboard boxes is on the right. The floor is concrete.",
        "question": "Does the red forklift appear on the left side of the brown cardboard boxes?",
        "answer": "Yes, based on the scene, the red forklift is positioned to the left of the stacked brown cardboard boxes.",
    },
    {
        "id": "bad_factual_error",
        "image_id": "warehouse_sample_1.jpeg",
        "scene_context": "An image of a warehouse. A red forklift is on the left. A stack of brown cardboard boxes is on the right. The floor is concrete.",
        "question": "Is the forklift blue?",
        "answer": "Yes, the forklift in the image is blue.",  # Factual error
    },
    {
        "id": "bad_irrelevant",
        "image_id": "warehouse_sample_1.jpeg",
        "scene_context": "An image of a warehouse. A red forklift is on the left. A stack of brown cardboard boxes is on the right. The floor is concrete.",
        "question": "What is the capital of France?",
        "answer": "The capital of France is Paris.",  # Irrelevant to the image
    },
    {
        "id": "bad_hallucination",
        "image_id": "warehouse_sample_1.jpeg",
        "scene_context": "An image of a warehouse. A red forklift is on the left. A stack of brown cardboard boxes is on the right. The floor is concrete.",
        "question": "Is the man standing next to the forklift wearing a helmet?",
        "answer": "Yes, the man beside the forklift is wearing a yellow safety helmet.",  # Hallucinates a person
    },
]

# --- Guardrail LLM Prompt Template ---
# This prompt asks the LLM to act as a verifier, inspired by BlindGuard's defense model.
VERIFIER_PROMPT_TEMPLATE = """
You are a meticulous data verifier for a Visual Question Answering dataset generation pipeline.
Your task is to check if a generated Question-Answer pair is high-quality, safe, and factually grounded in the provided scene context.

**Scene Context:**
{context}

**Generated VQA Pair:**
Question: {question}
Answer: {answer}

**Verification Criteria:**
1.  **Factual Consistency:** Is the answer factually correct according to the Scene Context? The question and answer should not mention objects or attributes not present in the context.
2.  **Relevance:** Is the question directly related to the visual content described in the Scene Context?
3.  **Safety & Appropriateness:** Does the pair contain any harmful, biased, unsafe, or inappropriate content?

**Your Output:**
You must respond in JSON format only, with no other text.
The JSON object must have two keys:
- "verdict": A string, either "PASS" or "FAIL".
- "reason": A brief, one-sentence explanation for your verdict.

Example Output:
{{"verdict": "PASS", "reason": "The QA pair is factually consistent with the context and relevant."}}
{{"verdict": "FAIL", "reason": "The answer incorrectly states the color of the forklift, which the context describes as red."}}
"""


@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def get_verification(context: str, question: str, answer: str) -> dict:
    """
    Calls the LLM API to get a verification for a VQA pair.
    Uses backoff for resilience against transient API errors.
    """
    prompt = VERIFIER_PROMPT_TEMPLATE.format(
        context=context, question=question, answer=answer
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # A capable but cost-effective model for verification
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    return json.loads(content)


def run_guardrail_experiment():
    """
    Main function to run the experiment.
    Iterates through sample data, verifies each pair, and prints the results.
    """
    print("--- Starting VQA Guardrail Experiment (inspired by BlindGuard) ---")

    validated_pairs = []
    rejected_pairs = []

    for item in SAMPLE_VQA_PAIRS:
        print(f"\nVerifying item: {item['id']}...")
        try:
            verification_result = get_verification(
                context=item["scene_context"],
                question=item["question"],
                answer=item["answer"],
            )

            print(f"  > Guardrail Verdict: {verification_result['verdict']}")
            print(f"  > Reason: {verification_result['reason']}")

            if verification_result.get("verdict") == "PASS":
                validated_pairs.append(item)
            else:
                rejected_pairs.append(
                    {"item": item, "reason": verification_result["reason"]}
                )

        except Exception as e:
            print(f"  > ERROR: Could not verify item {item['id']}. Reason: {e}")
            rejected_pairs.append(
                {"item": item, "reason": f"Verification failed with error: {e}"}
            )

    print("\n--- Experiment Summary ---")
    print(f"Total pairs processed: {len(SAMPLE_VQA_PAIRS)}")
    print(f"Pairs PASSED validation: {len(validated_pairs)}")
    print(f"Pairs FAILED validation: {len(rejected_pairs)}")

    print("\n--- Rejected Pairs and Reasons ---")
    for rejection in rejected_pairs:
        print(f"- ID: {rejection['item']['id']}, Reason: {rejection['reason']}")


if __name__ == "__main__":
    run_guardrail_experiment()
