# Copyright 2025 Garena Online Private Limited.
# Adapted from https://github.com/ZhangXJ199/EDGE-GRPO/blob/main/evaluate_model.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import time
import fire
import vllm
from datasets import load_dataset


def apply_vqa_prompt_template(question: str):
    """A placeholder for a VQA-specific prompt template (e.g., LLaVA)."""
    # Example for a LLaVA-like model, adapt as needed.
    return f"USER: <image>\n{question} ASSISTANT:"


def vqa_score_fn(pred: str, gt: str) -> float:
    """
    A simple scoring function for VQA.
        This is a placeholder and can be replaced with more sophisticated metrics.
    """
    return 1.0 if pred.strip().lower() == gt.strip().lower() else 0.0


# Example of a more complex grader from the source repo for math reasoning.
# from trl_edge.rewards.math_grader import boxed_reward_fn


def main(
    model_path: str,
    dataset_path: str,
    output_file: str = "results.json",
    dataset_split: str = "test",
    question_column: str = "question",
    answer_column: str = "answer",
    max_samples: int = None,
    tensor_parallel_size: int = 1,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
):
    """
    Runs evaluation on a VQA dataset using a vLLM model.

    Args:
        model_path: Path to the model on Hugging Face Hub or locally.
        dataset_path: Path to the dataset on Hugging Face Hub or locally.
        output_file: Path to save the JSON results.
        dataset_split: The split of the dataset to use (e.g., 'test', 'validation').
        question_column: The name of the column containing the question text.
        answer_column: The name of the column containing the ground truth answer.
        max_samples: Maximum number of samples to evaluate.
        tensor_parallel_size: Number of GPUs to use for tensor parallelism.
        temperature: The temperature for sampling.
        top_p: The top-p value for sampling.
        max_tokens: The maximum number of tokens to generate.
    """
    print(f"Loading model: {model_path}")
    llm = vllm.LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
    )

    print(f"Loading dataset: {dataset_path} (split: {dataset_split})")
    dataset = load_dataset(dataset_path, split=dataset_split)
    if max_samples is not None:
        dataset = dataset.select(range(max_samples))

    questions = [sample[question_column] for sample in dataset]
    ground_truths = [str(sample[answer_column]) for sample in dataset]
    prompts = [apply_vqa_prompt_template(q) for q in questions]

    print(f"Generating responses for {len(prompts)} samples...")
    sampling_params = vllm.SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_tokens
    )

    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()

    print(f"Generation took {end_time - start_time:.2f} seconds.")

    results = []
    total_score = 0
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text.strip()
        gt = ground_truths[i]
        score = vqa_score_fn(pred=generated_text, gt=gt)
        total_score += score

        results.append(
            {
                "question": questions[i],
                "prompt": prompt,
                "generated_answer": generated_text,
                "ground_truth_answer": gt,
                "score": score,
            }
        )

    accuracy = total_score / len(results) if results else 0
    print(f"Final Accuracy: {accuracy:.4f}")

    with open(output_file, "w") as f:
        json.dump({"results": results, "accuracy": accuracy}, f, indent=2)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
