import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Attempt to import required packages and provide helpful error messages
try:
    import nltk
    from rouge_score import rouge_scorer
except ImportError:
    logging.error("Dependencies not found. Please run: pip install nltk rouge-score")
    logging.error(
        "You may also need to download the NLTK sentence tokenizer: python -c \"import nltk; nltk.download('punkt')\""
    )
    exit(1)

# Download NLTK data if not present
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    logging.info("Downloading NLTK 'punkt' model for sentence tokenization...")
    nltk.download("punkt")


def load_nuggets(nugget_file: Path) -> List[str]:
    """Loads ground-truth nuggets from a .jsonl file."""
    nuggets = []
    with open(nugget_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if "nugget" in data and isinstance(data["nugget"], str):
                    nuggets.append(data["nugget"])
                else:
                    logging.warning(
                        f"Skipping invalid line in nuggets file: {line.strip()}"
                    )
            except json.JSONDecodeError:
                logging.warning(f"Could not decode JSON from line: {line.strip()}")
    logging.info(f"Loaded {len(nuggets)} nuggets from {nugget_file}")
    return nuggets


def load_report(report_file: Path) -> str:
    """Loads the generated report from a text file."""
    content = report_file.read_text(encoding="utf-8")
    logging.info(f"Loaded report of {len(content)} characters from {report_file}")
    return content


class ReportEvaluator:
    """
    Evaluates a generated report against a set of ground-truth nuggets.
    This implementation is inspired by the 'Report Quality' metrics from the RAVine framework.
    It uses ROUGE-L to measure similarity between nuggets and report sentences,
    serving as a lightweight proxy for RAVine's LLM-based evaluation.
    """

    def __init__(self, similarity_threshold: float = 0.5):
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self.threshold = similarity_threshold

    def evaluate(self, report_text: str, nuggets: List[str]) -> Dict[str, float]:
        """Calculates precision, recall, and completeness (F1 score) for the report."""
        if not nuggets:
            logging.warning("Nugget list is empty. Cannot compute metrics.")
            return {"precision": 0.0, "recall": 0.0, "completeness_f1": 0.0}

        report_sentences = nltk.sent_tokenize(report_text)
        if not report_sentences:
            logging.warning("Report is empty. Cannot compute metrics.")
            return {"precision": 0.0, "recall": 0.0, "completeness_f1": 0.0}

        nuggets_covered = [False] * len(nuggets)
        sentences_that_cover_nuggets = [False] * len(report_sentences)

        for i, nugget in enumerate(nuggets):
            for j, sentence in enumerate(report_sentences):
                score = self.scorer.score(nugget, sentence)["rougeL"].fmeasure
                if score >= self.threshold:
                    nuggets_covered[i] = True
                    sentences_that_cover_nuggets[j] = True
                    # A nugget can be covered by multiple sentences, but we only need one.
                    # A sentence can cover multiple nuggets.

        num_nuggets_covered = sum(nuggets_covered)
        num_sentences_covering = sum(sentences_that_cover_nuggets)

        recall = num_nuggets_covered / len(nuggets) if nuggets else 0.0
        precision = (
            num_sentences_covering / len(report_sentences) if report_sentences else 0.0
        )

        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        return {"precision": precision, "recall": recall, "completeness_f1": f1_score}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a generated report against ground-truth nuggets, inspired by RAVine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        required=True,
        help="Path to the model-generated report (.txt file).",
    )
    parser.add_argument(
        "--nuggets-file",
        type=Path,
        required=True,
        help="Path to the ground-truth nuggets (.jsonl file).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="ROUGE-L F1 score threshold to consider a nugget 'covered'.",
    )

    args = parser.parse_args()

    if not args.report_file.is_file():
        logging.error(f"Report file not found: {args.report_file}")
        return
    if not args.nuggets_file.is_file():
        logging.error(f"Nuggets file not found: {args.nuggets_file}")
        return

    # Load data
    report = load_report(args.report_file)
    nuggets = load_nuggets(args.nuggets_file)

    # Evaluate
    evaluator = ReportEvaluator(similarity_threshold=args.threshold)
    metrics = evaluator.evaluate(report, nuggets)

    # Print results
    print("\n--- Report Quality Evaluation Results ---")
    print(f"Precision:       {metrics['precision']:.4f}")
    print(f"Recall:          {metrics['recall']:.4f}")
    print(f"Completeness (F1): {metrics['completeness_f1']:.4f}")
    print("-----------------------------------------")


if __name__ == "__main__":
    main()
