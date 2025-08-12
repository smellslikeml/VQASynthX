import json
import argparse
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_rubric(item):
    """
    Performs a simplified rubric-based evaluation on the Chain-of-Thought reasoning.
    This mimics the 'Key Insight Extraction' and 'Coverage' from ResearcherBench.

    Args:
        item (dict): A single VQA item with 'question' and 'answer' keys.

    Returns:
        float: A score from 0.0 to 1.0.
    """
    score = 0
    max_score = 3.0
    reasoning = item.get("answer", "")
    question = item.get("question", "")

    # 1. Does the reasoning mention key terms from the question?
    question_keywords = [word for word in question.lower().split() if len(word) > 3 and word not in ["what", "how", "the", "is", "in", "from"]]
    if any(keyword in reasoning.lower() for keyword in question_keywords):
        score += 1
        logging.info(f"Item {item.get('id', 'N/A')}: [Rubric] Passed - Reasoning mentions question keywords.")
    else:
        logging.warning(f"Item {item.get('id', 'N/A')}: [Rubric] Failed - Reasoning does not mention keywords from question: {question_keywords}")

    # 2. Does the reasoning contain evidence of a spatial calculation or comparison?
    spatial_terms = ["feet", "meters", "close", "far", "left of", "right of", "behind", "in front of", "distance is"]
    if any(term in reasoning.lower() for term in spatial_terms):
        score += 1
        logging.info(f"Item {item.get('id', 'N/A')}: [Rubric] Passed - Reasoning includes spatial terms.")
    else:
        logging.warning(f"Item {item.get('id', 'N/A')}: [Rubric] Failed - Reasoning lacks explicit spatial terms.")

    # 3. Is there a clear conclusion stated?
    conclusion_phrases = ["the answer is", "therefore, the distance is", "in conclusion"]
    if any(phrase in reasoning.lower() for phrase in conclusion_phrases):
        score += 1
        logging.info(f"Item {item.get('id', 'N/A')}: [Rubric] Passed - Reasoning contains a clear conclusion phrase.")
    else:
        logging.warning(f"Item {item.get('id', 'N/A')}: [Rubric] Failed - Reasoning lacks a clear conclusion phrase.")

    return score / max_score

def evaluate_factual(item, ground_truth):
    """
    Performs a simplified factual evaluation.
    This mimics the 'Factual Assessment' from ResearcherBench by checking a claim
    against a source of truth. Here, the source is the simulated ground truth data.

    Args:
        item (dict): A single VQA item.
        ground_truth (dict): The ground truth data for this item.

    Returns:
        float: A score of 1.0 for faithful, 0.0 for unfaithful.
    """
    reasoning = item.get("answer", "")
    
    # 1. Extract claim from the reasoning text.
    try:
        claim_str = reasoning.split("The answer is ")[-1].split()[0]
        claim_value = float(claim_str)
    except (IndexError, ValueError):
        logging.warning(f"Item {item.get('id', 'N/A')}: [Factual] Failed - Could not extract a numerical claim from the answer.")
        return 0.0

    # 2. Verify claim against ground truth
    gt_value = ground_truth.get("distance_meters")
    # 1 meter = 3.28084 feet
    gt_value_feet = gt_value * 3.28084
    
    # Allow a 10% tolerance for estimation
    tolerance = 0.10 * gt_value_feet
    
    if abs(claim_value - gt_value_feet) <= tolerance:
        logging.info(f"Item {item.get('id', 'N/A')}: [Factual] Passed - Claim {claim_value:.2f} is within 10% tolerance of ground truth {gt_value_feet:.2f}.")
        return 1.0
    else:
        logging.warning(f"Item {item.get('id', 'N/A')}: [Factual] Failed - Claim {claim_value:.2f} is outside 10% tolerance of ground truth {gt_value_feet:.2f}.")
        return 0.0

def main():
    parser = argparse.ArgumentParser(description="Evaluate VQASynth data quality using ResearcherBench principles.")
    parser.add_argument("--input", type=str, required=True, help="Path to the generated VQA JSON file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the evaluation results JSON file.")
    args = parser.parse_args()

    logging.info(f"Loading data from {args.input}")
    with open(args.input, 'r') as f:
        data = json.load(f)

    # In a real pipeline, this would come from the depth/3D reconstruction stage.
    # Here, we simulate it for demonstration purposes.
    ground_truth_data = {
        item['id']: {"distance_meters": random.uniform(1.0, 10.0)} for item in data
    }

    results = []
    for item in data:
        item_id = item.get('id')
        if not item_id:
            logging.error("Item found without an 'id'. Skipping.")
            continue
        
        ground_truth = ground_truth_data.get(item_id)
        if not ground_truth:
            logging.error(f"No ground truth found for item id {item_id}. Skipping.")
            continue

        rubric_score = evaluate_rubric(item)
        factual_score = evaluate_factual(item, ground_truth)
        
        results.append({
            "id": item_id,
            "question": item.get("question"),
            "rubric_score": rubric_score,
            "factual_score": factual_score,
            "overall_score": (rubric_score + factual_score) / 2
        })

    logging.info(f"Saving evaluation results to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    logging.info("Evaluation complete.")

if __name__ == "__main__":
    main()
