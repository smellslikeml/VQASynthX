import re
import json
import argparse
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Inspired by the sensitive data patterns targeted by hawaii-pique-cwe200 (e.g., CWE-540, CWE-532),
# we define regex patterns to detect potential PII in generated text.
PII_PATTERNS = {
    "EMAIL": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "PHONE_NUMBER": re.compile(r"(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "CREDENTIALS_KEYWORDS": re.compile(
        r"\b(password|secret|key|token|username|credential|passwd)\b", re.IGNORECASE
    ),
}


def scan_text_for_pii(text: str) -> List[str]:
    """Scans a string for different types of PII patterns."""
    found_pii = []
    if not isinstance(text, str):
        return found_pii

    for pii_type, pattern in PII_PATTERNS.items():
        if pattern.search(text):
            found_pii.append(pii_type)
    return found_pii


def filter_vqa_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filters a list of VQA samples, removing any that contain PII in their
    'question' or 'answer' fields.
    """
    clean_data = []
    filtered_count = 0
    for item in data:
        # Assuming VQA data can have 'question'/'answer' or LLaVA-style 'conversations'.
        text_to_scan = []
        if "question" in item and item["question"]:
            text_to_scan.append(item["question"])
        if "answer" in item and item["answer"]:
            text_to_scan.append(item["answer"])
        if "conversations" in item:  # Handle LLaVA format
            for turn in item.get("conversations", []):
                if "value" in turn:
                    text_to_scan.append(turn["value"])

        found_pii_types = []
        for text in text_to_scan:
            found_pii_types.extend(scan_text_for_pii(text))

        if not found_pii_types:
            clean_data.append(item)
        else:
            filtered_count += 1
            logging.warning(
                f"Filtered item due to PII detection ({', '.join(set(found_pii_types))}). Content sample: {str(item)[:200]}..."
            )

    logging.info(
        f"Processing complete. Kept {len(clean_data)} items, filtered {filtered_count} items."
    )
    return clean_data


def main():
    """Main function to parse arguments and run the PII filter."""
    parser = argparse.ArgumentParser(
        description="Scans VQA-style JSON data to filter out entries containing potential PII. "
        "This is inspired by static analysis tools for sensitive data exposure."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input JSON file containing VQA data.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the filtered, PII-free JSON data.",
    )
    args = parser.parse_args()

    logging.info(f"Loading data from {args.input}...")
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            input_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logging.error(f"Failed to read or parse input file: {e}")
        return

    filtered_data = filter_vqa_data(input_data)

    logging.info(f"Writing filtered data to {args.output}...")
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(filtered_data, f, indent=2)
    except IOError as e:
        logging.error(f"Failed to write to output file: {e}")

    logging.info("PII filtering script finished.")


if __name__ == "__main__":
    main()
