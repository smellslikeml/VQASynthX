import json
import argparse
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def assess_conversation_trust(conversation: List[Dict[str, str]]) -> (float, str):
    """
    Assesses the "trust" of a single VQA conversation based on heuristics.
    This function is inspired by the "AI-Powered Trust Assessor" from the
    FortifyingAgenticWeb project. Instead of agent actions, we evaluate the
    quality and plausibility of synthetic VQA data.

    Args:
        conversation: A list of dicts, representing turns in a conversation.
                      Expected format: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]

    Returns:
        A tuple containing a trust score (0.0 to 1.0) and a justification string.
    """
    if not isinstance(conversation, list) or len(conversation) < 2:
        return 0.0, "Invalid conversation format."

    question = conversation[0].get("value", "")
    answer = conversation[1].get("value", "")

    if not question or not answer:
        return 0.0, "Missing question or answer value."

    justifications = []
    score = 1.0

    # Rule 1: Check for emptiness or very short content
    if len(question.split()) < 3 or len(answer.split()) < 2:
        score -= 0.8
        justifications.append("Question or answer is too short.")

    # Rule 2: Check for placeholder text from generation templates
    placeholders = [
        "placeholder",
        "not specified",
        "N/A",
        "[object]",
        "describe the image",
    ]
    if any(p in question.lower() for p in placeholders) or any(
        p in answer.lower() for p in placeholders
    ):
        score -= 0.9
        justifications.append("Contains placeholder text.")

    # Rule 3: Check if question and answer are nearly identical
    if question.strip().lower() == answer.strip().lower():
        score -= 0.7
        justifications.append("Question and answer are identical.")

    # Rule 4: Simple check for "I don't know" or refusal answers which might indicate generation failure
    refusals = ["i cannot", "i can't", "i do not know", "unable to answer"]
    if any(r in answer.lower() for r in refusals):
        score -= 0.5
        justifications.append("Answer contains a refusal.")

    final_score = max(0.0, score)
    return final_score, " ".join(justifications) if justifications else "High trust"


def main():
    """
    Main function to run the data assessment pipeline.
    This script introduces a data quality/security stage to the VQASynth process,
    inspired by the security-focused architecture of FortifyingAgenticWeb.
    """
    parser = argparse.ArgumentParser(
        description="Assess and filter synthetic VQA data for quality and trustworthiness."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input JSONL file containing VQA conversation data.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to write the filtered, high-trust JSONL data.",
    )
    parser.add_argument(
        "--trust_threshold",
        type=float,
        default=0.7,
        help="Minimum trust score to keep a VQA pair.",
    )
    args = parser.parse_args()

    logging.info(f"Starting data trust assessment for file: {args.input_path}")
    logging.info(f"Trust threshold set to: {args.trust_threshold}")

    processed_count = 0
    kept_count = 0

    try:
        with open(args.input_path, "r", encoding="utf-8") as infile, open(
            args.output_path, "w", encoding="utf-8"
        ) as outfile:
            for line in infile:
                processed_count += 1
                try:
                    data = json.loads(line)
                    conversation = data.get("conversations")

                    if conversation is None:
                        logging.warning(
                            f"Skipping line with missing 'conversations' key: {line.strip()}"
                        )
                        continue

                    score, justification = assess_conversation_trust(conversation)

                    if score >= args.trust_threshold:
                        outfile.write(json.dumps(data) + "\n")
                        kept_count += 1
                    else:
                        logging.debug(
                            f"Filtered out item with score {score:.2f}: Reason: {justification}"
                        )

                except json.JSONDecodeError:
                    logging.warning(f"Skipping invalid JSON line: {line.strip()}")
                except Exception as e:
                    logging.error(f"Error processing line: {line.strip()} - {e}")
    except FileNotFoundError:
        logging.error(f"Input file not found: {args.input_path}")
        return

    logging.info("Assessment complete.")
    logging.info(f"Total items processed: {processed_count}")
    logging.info(f"Items kept (passed trust threshold): {kept_count}")
    if processed_count > 0:
        filtered_count = processed_count - kept_count
        filter_rate = (filtered_count / processed_count) * 100
        logging.info(f"Items filtered out: {filtered_count} ({filter_rate:.2f}%)")
    logging.info(f"High-trust data saved to: {args.output_path}")


if __name__ == "__main__":
    main()
