import os
import json
import base64
import argparse
from openai import OpenAI

# Configuration
# For this minimal example, the API key is expected to be in the environment.
API_KEY = os.environ.get("OPENAI_API_KEY")
CLAIM_EXTRACTION_MODEL = "gpt-4-turbo"
VISUAL_JUDGE_MODEL = "gpt-4o"
CLIENT = OpenAI(api_key=API_KEY)

# Prompts inspired by ResearcherBench's claim/evidence verification
CLAIM_EXTRACTION_PROMPT = """
You are a claim extraction system. Your task is to analyze the provided text, which is an answer to a question about an image, and extract all distinct, verifiable factual claims. A claim is a statement of fact about the objects, their properties, or their spatial relationships in the image.

- Each claim must be a standalone statement.
- Do not extract questions or parts of the reasoning process that are not statements of fact.
- Focus on claims about positions, distances, counts, and orientations.
- Output the claims as a JSON list of strings. If no claims are found, return an empty list.

Here is the text to analyze:
---
{answer_text}
---

JSON list of claims:
"""

VISUAL_VERIFICATION_PROMPT = """
You are a visual verification system. Your task is to determine if a given factual claim about an image is supported by the visual evidence in that image.

Respond with a JSON object containing two keys:
1. "reasoning": A brief explanation of your decision, referencing specific parts of the image.
2. "supported": A boolean value, `true` if the claim is supported by the image, and `false` otherwise.

Here is the factual claim:
---
{claim}
---
"""

def encode_image_to_base64(image_path):
    """Encodes an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_claims(answer_text):
    """Uses an LLM to extract factual claims from a model's answer."""
    try:
        response = CLIENT.chat.completions.create(
            model=CLAIM_EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts factual claims and outputs them in a JSON list."},
                {"role": "user", "content": CLAIM_EXTRACTION_PROMPT.format(answer_text=answer_text)}
            ],
            response_format={"type": "json_object"}
        )
        content = json.loads(response.choices[0].message.content)
        # The model might return a dictionary with a 'claims' key or just a list.
        if isinstance(content, dict) and 'claims' in content and isinstance(content['claims'], list):
             return content['claims']
        elif isinstance(content, list):
             return content
        else:
            print(f"Warning: Unexpected format from claim extractor: {content}")
            return []

    except Exception as e:
        print(f"Error extracting claims: {e}")
        return []

def verify_claim_with_image(claim, image_base64):
    """Uses a VLM to verify if a claim is supported by an image."""
    try:
        response = CLIENT.chat.completions.create(
            model=VISUAL_JUDGE_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISUAL_VERIFICATION_PROMPT.format(claim=claim)},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=300
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"Error during visual verification for claim '{claim}': {e}")
        return {"reasoning": str(e), "supported": False}

def run_visual_faithfulness_eval(data_path):
    """
    Main function to run the visual faithfulness evaluation.
    This process is inspired by the Factual Assessment in ResearcherBench.
    """
    if not API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    with open(data_path, 'r') as f:
        vqa_data = json.load(f)

    total_claims = 0
    supported_claims = 0
    results_log = []

    for i, item in enumerate(vqa_data):
        print(f"\n--- Processing Item {i+1}/{len(vqa_data)} ---")
        image_path = item.get("image")
        answer = item.get("answer_cot")

        if not all([image_path, answer]):
            print(f"Skipping item {i+1} due to missing 'image' or 'answer_cot'.")
            continue
        
        print(f"Extracting claims from answer...")
        claims = extract_claims(answer)
        print(f"Found {len(claims)} claims.")

        if not claims:
            continue

        base64_image = encode_image_to_base64(image_path)
        
        item_results = {
            "item_id": i + 1,
            "image": image_path,
            "claims_evaluation": []
        }

        for claim in claims:
            total_claims += 1
            verification = verify_claim_with_image(claim, base64_image)
            is_supported = verification.get("supported", False)

            if is_supported:
                supported_claims += 1

            print(f"  - Claim: '{claim}' -> Supported: {is_supported}")
            
            item_results["claims_evaluation"].append({
                "claim": claim,
                "is_supported": is_supported,
                "judge_reasoning": verification.get("reasoning")
            })
        
        results_log.append(item_results)

    if total_claims == 0:
        faithfulness_score = 0.0
        print("\nNo claims were extracted from the provided data.")
    else:
        faithfulness_score = supported_claims / total_claims
        print(f"\nEvaluation Complete.")
        print(f"Total Claims: {total_claims}")
        print(f"Supported Claims: {supported_claims}")

    print(f"\nVisual Faithfulness Score: {faithfulness_score:.4f}")

    with open("visual_faithfulness_results.json", "w") as f:
        json.dump(results_log, f, indent=2)
    print("Detailed results saved to visual_faithfulness_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Visual Faithfulness Evaluation on VQA data.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the JSON file containing VQA data.")
    args = parser.parse_args()
    
    run_visual_faithfulness_eval(args.data_path)
