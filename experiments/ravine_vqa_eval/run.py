import json
import argparse
import pandas as pd


def load_jsonl(file_path):
    """Loads a JSONL file into a list of dictionaries."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        exit(1)


def evaluate_answer(llm_answer, ground_truth_nuggets):
    """
    Evaluates an LLM's answer against a list of ground-truth nuggets.
    This is a simplified, non-LLM version of RAVine's nugget-based evaluation.
    """
    if not ground_truth_nuggets:
        return {"matched_nuggets": 0, "total_gt_nuggets": 0}

    matched_nuggets = 0
    # Use simple, case-insensitive substring matching for this minimal example.
    # A more robust solution would use semantic similarity or an LLM judge.
    for nugget in ground_truth_nuggets:
        if nugget.lower() in llm_answer.lower():
            matched_nuggets += 1

    return {
        "matched_nuggets": matched_nuggets,
        "total_gt_nuggets": len(ground_truth_nuggets),
    }


def main(args):
    """
    Main function to run the evaluation.
    This script adapts the core idea of RAVine: evaluating agent performance
    based on the presence of ground-truth "nuggets" in the final output.
    Instead of web search, we evaluate VQA task performance on generated text.
    """
    print(f"Loading VQA logs from: {args.log_path}")
    vqa_logs = {item["query"]: item for item in load_jsonl(args.log_path)}

    print(f"Loading ground truth nuggets from: {args.gt_path}")
    ground_truth = {item["query"]: item for item in load_jsonl(args.gt_path)}

    results = []

    for query, gt_item in ground_truth.items():
        if query not in vqa_logs:
            print(f"Warning: Query '{query}' not found in logs. Skipping.")
            continue

        log_item = vqa_logs[query]
        llm_answer = log_item.get("llm_answer", "")
        gt_nuggets = gt_item.get("nuggets", [])

        eval_result = evaluate_answer(llm_answer, gt_nuggets)

        recall = (
            eval_result["matched_nuggets"] / eval_result["total_gt_nuggets"]
            if eval_result["total_gt_nuggets"] > 0
            else 0
        )
        # "Task Completion" is defined as answering with at least one correct nugget.
        task_completed = 1 if eval_result["matched_nuggets"] > 0 else 0

        results.append(
            {
                "query": query,
                "recall": recall,
                "task_completed": task_completed,
                "matched_nuggets": eval_result["matched_nuggets"],
                "total_gt_nuggets": eval_result["total_gt_nuggets"],
            }
        )

    if not results:
        print("No results to display. Check input files and ensure queries match.")
        return

    # Aggregate and display results in a summary table, similar to RAVine's output
    results_df = pd.DataFrame(results)

    avg_recall = results_df["recall"].mean() * 100
    task_completion_rate = results_df["task_completed"].mean() * 100

    print("\n--- VQA Performance Evaluation (inspired by RAVine) ---")
    print(f"\nOverall Metrics:")
    print(f"  Task Completion Rate (%): {task_completion_rate:.2f} (↑)")
    print(f"  Nugget Recall (%):        {avg_recall:.2f} (↑)")
    print("\n--- Detailed Results per Query ---")
    print(results_df.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate VQA logs using a RAVine-inspired nugget-based approach."
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="experiments/ravine_vqa_eval/sample_logs.jsonl",
        help="Path to the VQA log file (JSONL).",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default="experiments/ravine_vqa_eval/sample_nuggets.jsonl",
        help="Path to the ground truth nuggets file (JSONL).",
    )

    args = parser.parse_args()
    main(args)
