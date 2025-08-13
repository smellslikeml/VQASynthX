import argparse
import json
import logging
import os
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nltk
import numpy as np
import pandas as pd
import pytrec_eval
from huggingface_hub import hf_hub_download
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Data structures adapted from RAVine: src/data/types.py
@dataclass
class Nugget:
    nugget_id: str
    text: str
    source: str


@dataclass
class LogData:
    query: str
    query_id: str
    report: Optional[str] = None
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)
    search_results: List[List[Dict[str, Any]]] = field(default_factory=list)
    retrieved_urls: List[str] = field(default_factory=list)
    gt_urls: List[str] = field(default_factory=list)
    latency: Optional[float] = None


@dataclass
class EvalResult:
    report_eval: Dict[str, Any] = field(default_factory=dict)
    search_eval: Dict[str, Any] = field(default_factory=dict)
    fetch_eval: Dict[str, Any] = field(default_factory=dict)


class RavineLogEvaluator:
    """A self-contained evaluator inspired by RAVine's LogEvaluator."""

    def __init__(self, nuggets_path: str, qrels_path: str, mapper_path: str):
        logging.info("Initializing RavineLogEvaluator...")
        self.nuggets, self.qrels, self.url_to_docid = self._load_ground_truth(
            nuggets_path, qrels_path, mapper_path
        )
        self.eval_model = "gemini-1.5-flash-latest"  # Mocking this as we use pre-computed report evals
        logging.info("Ground truth data loaded.")

    def _load_ground_truth(self, nuggets_path, qrels_path, mapper_path):
        with open(nuggets_path, "r") as f:
            nuggets_data = [json.loads(line) for line in f]
        nuggets = {}
        for item in nuggets_data:
            nuggets[item["query_id"]] = [
                Nugget(nugget["nugget_id"], nugget["text"], nugget["source"])
                for nugget in item["nuggets"]
            ]

        with open(qrels_path, "r") as f:
            qrels_data = [json.loads(line) for line in f]
        qrels = {item["query_id"]: item["doc_ids"] for item in qrels_data}

        with open(mapper_path, "r") as f:
            url_to_docid = json.load(f)

        return nuggets, qrels, url_to_docid

    def parse_log(self, log_path: str) -> Optional[LogData]:
        try:
            with open(log_path, "r") as f:
                log_content = json.load(f)

            query_id = log_content.get("query_id", "qid_unknown")
            query = log_content.get("query", "query_unknown")
            report = log_content.get("output", "")
            latency = log_content.get("latency", 0)

            trajectory = log_content.get("trajectory", [])
            if not isinstance(trajectory, list):
                trajectory = []

            search_queries = []
            search_results = []
            retrieved_urls = set()

            for turn in trajectory:
                if isinstance(turn, dict) and turn.get("tool_name") == "search":
                    tool_input = turn.get("tool_input", {})
                    if isinstance(tool_input, dict):
                        search_queries.append(tool_input.get("query", ""))

                if (
                    isinstance(turn, dict)
                    and turn.get("tool_output")
                    and turn.get("tool_name") == "search"
                ):
                    tool_output = turn["tool_output"]
                    if isinstance(tool_output, str):
                        try:
                            tool_output = json.loads(tool_output)
                        except json.JSONDecodeError:
                            tool_output = []
                    search_results.append(tool_output)

            if log_content.get("retrieved_snippets"):
                for url in log_content["retrieved_snippets"]:
                    retrieved_urls.add(url)

            gt_urls = []
            if query_id in self.nuggets:
                gt_urls = list({n.source for n in self.nuggets[query_id]})

            return LogData(
                query=query,
                query_id=query_id,
                report=report,
                trajectory=trajectory,
                search_queries=search_queries,
                search_results=search_results,
                retrieved_urls=list(retrieved_urls),
                gt_urls=gt_urls,
                latency=latency,
            )
        except Exception as e:
            logging.error(f"Failed to parse log file {log_path}: {e}")
            return None

    def _calculate_report_quality(self, log_data: LogData) -> Dict[str, Any]:
        # In the original RAVine, this calls an LLM-as-a-judge.
        # Here, we extract the pre-computed scores from the log file itself.
        log_report_eval = log_data.trajectory[-1].get("report_eval", {})
        return {
            "Rate": log_report_eval.get("Task Completion Rate", 0.0),
            "Comp.": log_report_eval.get("Completeness", 0.0),
            "Rec.": log_report_eval.get("Recall", 0.0),
            "Prec.": log_report_eval.get("Precision", 0.0),
            "Latency": log_data.latency or 0.0,
        }

    def _calculate_search_performance(self, log_data: LogData) -> Dict[str, Any]:
        qrels = {
            self.qrels.get(log_data.query_id, {}).get(doc_id, 0)
            for doc_id in self.qrels.get(log_data.query_id, {})
        }
        qrels_dict = {log_data.query_id: self.qrels.get(log_data.query_id, {})}

        run = defaultdict(dict)
        for turn_idx, results in enumerate(log_data.search_results):
            for rank, doc in enumerate(results, 1):
                doc_id = doc.get("docid")
                if doc_id and doc_id not in run[log_data.query_id]:
                    run[log_data.query_id][doc_id] = 1 / rank

        evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, {"recall", "P"})
        results = evaluator.evaluate(run)
        query_res = results.get(log_data.query_id, {})

        return {
            "Prec@1": query_res.get("P_1", 0.0),
            "Prec@3": query_res.get("P_3", 0.0),
            "Recall@10": query_res.get("recall_10", 0.0),
            "Recall@100": query_res.get("recall_100", 0.0),
            "Turns": len(log_data.search_queries),
        }

    def _calculate_fetch_performance(self, log_data: LogData) -> Dict[str, Any]:
        retrieved_docids = {
            self.url_to_docid.get(url) for url in log_data.retrieved_urls
        }
        retrieved_docids.discard(None)

        gt_docids = {self.url_to_docid.get(url) for url in log_data.gt_urls}
        gt_docids.discard(None)

        url_errors = len(log_data.retrieved_urls) - len(retrieved_docids)

        if not retrieved_docids:
            return {"Prec.": 0.0, "Rec.": 0.0, "Gain": 0.0, "URL Err.": url_errors}

        precision = len(retrieved_docids.intersection(gt_docids)) / len(
            retrieved_docids
        )
        recall = (
            len(retrieved_docids.intersection(gt_docids)) / len(gt_docids)
            if gt_docids
            else 0.0
        )

        # Information Gain (simplified from RAVine)
        sentences = nltk.sent_tokenize(log_data.report or "")
        gain = len(sentences)

        return {
            "Prec.": precision,
            "Rec.": recall,
            "Gain": float(gain),
            "URL Err.": url_errors,
        }

    def evaluate(self, log_path: str) -> Optional[EvalResult]:
        log_data = self.parse_log(log_path)
        if not log_data:
            return None

        return EvalResult(
            report_eval=self._calculate_report_quality(log_data),
            search_eval=self._calculate_search_performance(log_data),
            fetch_eval=self._calculate_fetch_performance(log_data),
        )

    def run_evaluation_on_dir(self, log_dir: str) -> Dict[str, Any]:
        all_results = []
        log_files = sorted(list(Path(log_dir).glob("*.json")))
        logging.info(f"Found {len(log_files)} log files in {log_dir}.")

        for log_file in tqdm(log_files, desc="Evaluating Logs"):
            result = self.evaluate(str(log_file))
            if result:
                all_results.append(result)

        if not all_results:
            logging.warning("No results to aggregate.")
            return {}

        # Aggregate results
        agg = defaultdict(list)
        for res in all_results:
            for k, v in res.report_eval.items():
                agg[f"Report/{k}"].append(v)
            for k, v in res.search_eval.items():
                agg[f"Search/{k}"].append(v)
            for k, v in res.fetch_eval.items():
                agg[f"Fetch/{k}"].append(v)

        summary = {k: statistics.mean(v) for k, v in agg.items()}
        return summary


def download_ravine_data(data_dir: str, model_name: str):
    Path(data_dir).mkdir(exist_ok=True, parents=True)
    repo_files_to_download = {
        "RAVine-nuggets": "rag24.test.final.nuggets.jsonl",
        "RAVine-qrels": "qrels.jsonl",
        "RAVine-mapper": "url2doc.msmarco-v2.1-doc.json",
    }

    for repo_id, filename in repo_files_to_download.items():
        local_path = Path(data_dir) / filename
        if not local_path.exists():
            logging.info(f"Downloading {filename} from {repo_id}...")
            hf_hub_download(
                repo_id=f"sapphirex/{repo_id}",
                filename=filename,
                repo_type="dataset",
                local_dir=data_dir,
            )

    log_repo_id = "sapphirex/RAVine-logs"
    log_dir_name = model_name.replace("/", "_")  # Simplification
    log_dir_local = Path(data_dir) / model_name
    if not log_dir_local.exists():
        logging.info(f"Downloading logs for {model_name}...")
        # This downloads all files from a specific folder in the repo
        hf_hub_download(
            repo_id=log_repo_id,
            repo_type="dataset",
            local_dir=str(log_dir_local),
            allow_patterns=f"{model_name}/**",
            local_dir_use_symlinks=False,
        )
    # The files are downloaded into a nested dir, move them up
    nested_dir = log_dir_local / model_name
    if nested_dir.exists():
        for f in nested_dir.iterdir():
            f.rename(log_dir_local / f.name)
        nested_dir.rmdir()
    return log_dir_local


def main():
    parser = argparse.ArgumentParser(
        description="Run log-based evaluation using the RAVine benchmark."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Name of the model whose logs to evaluate (e.g., 'meta-llama/Llama-3.1-8B-Instruct').",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./ravine_data",
        help="Directory to store downloaded benchmark data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ravine_results",
        help="Directory to save the evaluation report.",
    )
    args = parser.parse_args()

    # Ensure NLTK data is available
    try:
        nltk.data.find("tokenizers/punkt")
    except nltk.downloader.DownloadError:
        logging.info("Downloading NLTK 'punkt' tokenizer...")
        nltk.download("punkt")

    # Download data and get path to logs
    log_dir = download_ravine_data(args.data_dir, args.model_name)

    # Initialize evaluator
    evaluator = RavineLogEvaluator(
        nuggets_path=os.path.join(args.data_dir, "rag24.test.final.nuggets.jsonl"),
        qrels_path=os.path.join(args.data_dir, "qrels.jsonl"),
        mapper_path=os.path.join(args.data_dir, "url2doc.msmarco-v2.1-doc.json"),
    )

    # Run evaluation
    summary_report = evaluator.run_evaluation_on_dir(str(log_dir))

    # Save and print report
    Path(args.output_dir).mkdir(exist_ok=True)
    output_path = os.path.join(
        args.output_dir, f"{args.model_name.replace('/', '_')}_report.json"
    )
    with open(output_path, "w") as f:
        json.dump(summary_report, f, indent=2)

    logging.info(f"Evaluation report saved to {output_path}")

    # Print a formatted summary
    print("\n--- RAVine Evaluation Summary ---")
    print(f"Model: {args.model_name}")
    print("-----------------------------------")
    df = pd.DataFrame.from_dict(
        summary_report, orient="index", columns=["Average Score"]
    )
    df.index.name = "Metric"
    print(df.round(4))
    print("-----------------------------------")


if __name__ == "__main__":
    main()
