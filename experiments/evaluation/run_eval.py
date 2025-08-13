import os
import json
import random
import time
from prompts import JUDGE_PROMPT

# Configuration for the Judging LLM, inspired by EESE's config.py
# For this test, we will use mock functions and not a real API.
JUDGE_LLM_CONFIG = {
    "api_key": os.environ.get("OPENAI_API_KEY", "sk-mock-key-for-testing"),
    "model": "gpt-4o",
    "max_retries": 3,
}

# A mini-benchmark for spatial reasoning evaluation
SPATIAL_BENCHMARK = [
    {
        "id": "sample-001",
        "image_context": "A living room with a red sofa on the left and a blue chair on the right.",
        "question": "Is the red sofa to the left or right of the blue chair?",
        "ground_truth_answer": "The red sofa is to the left of the blue chair.",
    },
    {
        "id": "sample-002",
        "image_context": "A desk with a laptop placed approximately 2 feet in front of a monitor.",
        "question": "How far is the laptop from the monitor?",
        "ground_truth_answer": "The laptop is about 2 feet from the monitor.",
    },
    {
        "id": "sample-003",
        "image_context": "A kitchen scene where a green apple is on the counter, and a red apple is in a bowl on the table.",
        "question": "Which apple is closer to the floor?",
        "ground_truth_answer": "The red apple in the bowl on the table is likely higher and therefore further from the floor than the green apple on the counter, but this depends on the height of the counter vs the table.",
    },
]

MOCK_MODEL_ANSWERS = {
    "sample-001": "The sofa, which is red, is on the left side.",
    "sample-002": "It is around 2 feet away.",
    "sample-003": "The one on the counter.",
}


def call_model_under_test(sample):
    """Mock function to simulate a call to the model being evaluated."""
    print(f"INFO: Getting response from model under test for sample {sample['id']}")
    time.sleep(0.1)  # Simulate network latency
    return MOCK_MODEL_ANSWERS.get(sample["id"], "I don't know.")


def call_judge_llm(prompt, model_name, max_retries=3):
    """Mock function to simulate a call to the judging LLM, inspired by EESE's call.py."""
    print(f"INFO: Calling Judge LLM ({model_name}) to score the response.")
    for attempt in range(max_retries):
        try:
            # This simulates the random choice from EESE's mock implementation
            score_str = random.choice(["7", "8", "9", "10", "bad_response"])
            if score_str == "bad_response":
                raise ValueError("Simulated API error")

            score = int(score_str)
            print(f"INFO: Judge LLM returned score: {score}")
            return score
        except (ValueError, TypeError) as e:
            print(
                f"WARNING: Judge LLM returned invalid score on attempt {attempt + 1}. Retrying..."
            )
            time.sleep(0.2)
    print("ERROR: Judge LLM failed to return a valid score after max retries.")
    return 0  # Return a default score on failure


def main():
    """Main function to run the evaluation benchmark."""
    print("Starting Spatial Reasoning Evaluation...")
    results = []
    total_score = 0

    for sample in SPATIAL_BENCHMARK:
        print(f"\n--- Evaluating Sample: {sample['id']} ---")
        model_answer = call_model_under_test(sample)

        judge_prompt = JUDGE_PROMPT.format(
            question=sample["question"],
            ground_truth_answer=sample["ground_truth_answer"],
            model_answer=model_answer,
        )

        score = call_judge_llm(
            prompt=judge_prompt,
            model_name=JUDGE_LLM_CONFIG["model"],
            max_retries=JUDGE_LLM_CONFIG["max_retries"],
        )

        results.append(
            {
                "id": sample["id"],
                "question": sample["question"],
                "ground_truth_answer": sample["ground_truth_answer"],
                "model_answer": model_answer,
                "score": score,
            }
        )
        total_score += score

    print("\n--- Evaluation Complete ---")
    print("\nDetailed Results:")
    print(json.dumps(results, indent=2))

    average_score = total_score / len(SPATIAL_BENCHMARK) if SPATIAL_BENCHMARK else 0
    print(f"\nSummary:")
    print(f"  - Total Samples: {len(SPATIAL_BENCHMARK)}")
    print(f"  - Average Score: {average_score:.2f} / 10")


if __name__ == "__main__":
    main()
