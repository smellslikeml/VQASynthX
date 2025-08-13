import argparse
import json
import os
import re
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# Regex to extract the score from the model's output
SCORE_PATTERN = re.compile(r"score:\s*\{(\d+)\}")


class VLMJudgeFilter:
    """
    Uses a compact VLM to judge the quality and alignment of image-caption pairs.
    This class is an adaptation of the methodology from the "Compact_VLM_Filter" repository.
    """

    def __init__(self, model_path: str, batch_size: int = 32, device: str = "cuda"):
        self.batch_size = batch_size
        self.device = device

        print(f"Loading VLM judge model from {model_path}...")
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, device_map=self.device, torch_dtype=torch.float16
        )
        self.model.eval()
        print("VLM judge model loaded successfully.")

    def _create_prompts(self, captions: list[str]) -> list[str]:
        """Creates the batch of prompts for the VLM judge."""
        # This prompt structure is inferred from the fine-tuning of the judge model.
        # It asks for a structured score output.
        prompt_template = "Is the caption appropriate for the image? Please give me a score from 1 to 10. score: {{score}}"

        messages = []
        for caption in captions:
            # The structure <|im_start|><image><|im_end|> is handled by the processor
            # when text and images are passed together. We just need the text part.
            text_content = f"{caption}\n{prompt_template}"
            messages.append(text_content)
        return messages

    def get_scores(self, image_paths: list[str], captions: list[str]) -> list[int]:
        """
        Processes a batch of images and captions to get alignment scores.
        Returns a list of scores, with -1 indicating an error for that item.
        """
        scores = [-1] * len(image_paths)
        try:
            images = [Image.open(p).convert("RGB") for p in image_paths]
            prompts = self._create_prompts(captions)

            inputs = self.processor(
                text=prompts, images=images, return_tensors="pt", padding=True
            ).to(self.device)

            with torch.no_grad():
                gen_kwargs = {"max_new_tokens": 20}
                outputs = self.model.generate(**inputs, **gen_kwargs)

            # Decode and parse scores
            responses = self.processor.batch_decode(outputs, skip_special_tokens=True)
            for i, response in enumerate(responses):
                match = SCORE_PATTERN.search(response)
                if match:
                    scores[i] = int(match.group(1))
                else:
                    print(f"Warning: Could not parse score from response: '{response}'")
        except Exception as e:
            print(f"Error processing batch: {e}")
            # All items in this batch will have a score of -1

        return scores

    def filter_dataset(self, data: list[dict], score_threshold: int) -> list[dict]:
        """
        Filters a dataset based on the VLM judge's scores.
        Assumes data is a list of dicts, each with "image_path" and "caption".
        """
        filtered_data = []

        # Process data in batches
        for i in tqdm(
            range(0, len(data), self.batch_size), desc="Filtering with VLM Judge"
        ):
            batch = data[i : i + self.batch_size]
            image_paths = [item["image_path"] for item in batch]
            captions = [item["caption"] for item in batch]

            scores = self.get_scores(image_paths, captions)

            for item, score in zip(batch, scores):
                if score >= score_threshold:
                    item["quality_score"] = score  # Add score to the output data
                    filtered_data.append(item)
                else:
                    print(
                        f"Filtering out item with score {score}: {item['image_path']}"
                    )

        return filtered_data


def main():
    parser = argparse.ArgumentParser(
        description="Filter image-caption datasets using a VLM Judge."
    )
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the input JSON file."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the filtered output JSON file.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Dauka-transformers/Compact_VLM_filter",
        help="Path or HF repo for the VLM judge model.",
    )
    parser.add_argument(
        "--score_threshold",
        type=int,
        default=5,
        help="Minimum score (inclusive) to keep an item.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for processing."
    )

    args = parser.parse_args()

    print(f"Loading data from: {args.input_path}")
    try:
        with open(args.input_path, "r") as f:
            input_data = json.load(f)
    except Exception as e:
        print(f"Error loading input data: {e}")
        return

    if not isinstance(input_data, list) or not all(
        "image_path" in d and "caption" in d for d in input_data
    ):
        print(
            "Invalid input data format. Expected a JSON list of objects, each with 'image_path' and 'caption'."
        )
        return

    vlm_filter = VLMJudgeFilter(model_path=args.model_path, batch_size=args.batch_size)
    filtered_data = vlm_filter.filter_dataset(input_data, args.score_threshold)

    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.output_path, "w") as f:
        json.dump(filtered_data, f, indent=2)

    print(
        f"Filtering complete. Original items: {len(input_data)}, Filtered items: {len(filtered_data)}"
    )
    print(f"Filtered data saved to: {args.output_path}")


if __name__ == "__main__":
    main()
