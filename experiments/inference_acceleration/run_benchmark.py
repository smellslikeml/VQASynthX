import argparse
import time
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import requests
import warnings
import os


# --- Logic adapted from LessIsMore/src/enable_tidal.py and tidal_build ---
# This is a simplified, self-contained representation of the patching process.
# In a real implementation, this would likely involve importing from the LessIsMore source.
def enable_less_is_more(model, budget=1024, recent_ratio=0.25):
    """
    Monkey-patches the model to use LessIsMore sparse attention.
    This is a conceptual adaptation. The actual LessIsMore library
    should be installed and its patching functions used.
    """
    print(
        f"Attempting to enable LessIsMore with budget={budget} and recent_ratio={recent_ratio}"
    )
    try:
        # The actual library provides a cleaner way to do this.
        # This simulates the patching process for a Llama-based model like LLaVA.
        from src.tidal_build.modify_llama_lim import convert_to_lim_attn

        model = convert_to_lim_attn(model, budget, recent_ratio)
        print("Successfully patched model with LessIsMore attention.")
        # Set environment variables that the custom kernels might use
        os.environ["TIDAL_K"] = str(budget)
        os.environ["TIDAL_R"] = str(recent_ratio)
    except ImportError:
        warnings.warn(
            "Could not import LessIsMore patching functions. "
            "Ensure the LessIsMore library is correctly installed. "
            "Running with standard attention."
        )
    except Exception as e:
        warnings.warn(
            f"Failed to apply LessIsMore patch: {e}. Running with standard attention."
        )

    return model


def run_benchmark(model, processor, image_url, question, max_new_tokens=100):
    """Runs a single inference pass and measures latency."""
    raw_image = Image.open(requests.get(image_url, stream=True).raw)
    prompt = f"USER: <image>\n{question} ASSISTANT:"

    inputs = processor(text=prompt, images=raw_image, return_tensors="pt").to("cuda:0")

    start_time = time.time()

    # generate response
    with torch.no_grad():
        generation_output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )

    end_time = time.time()

    latency = end_time - start_time
    generated_ids = generation_output[0][inputs.input_ids.shape[1] :]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    num_tokens = len(generated_ids)

    return response, latency, num_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark VQA inference with and without LessIsMore."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="Hugging Face model ID for the LLaVA model.",
    )
    parser.add_argument(
        "--enable_less_is_more",
        action="store_true",
        help="Enable LessIsMore sparse attention.",
    )
    parser.add_argument(
        "--budget", type=int, default=2048, help="Token budget for LessIsMore."
    )
    parser.add_argument(
        "--recent_ratio",
        type=float,
        default=0.25,
        help="Ratio of budget reserved for recent tokens in LessIsMore.",
    )
    args = parser.parse_args()

    # --- Load Model and Processor ---
    print(f"Loading model: {args.model_id}")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda:0")
    processor = AutoProcessor.from_pretrained(args.model_id)

    # --- Apply LessIsMore if enabled ---
    if args.enable_less_is_more:
        model = enable_less_is_more(model, args.budget, args.recent_ratio)

    # --- Benchmark Data ---
    # Using a sample from the VQASynth README
    test_case = {
        "image_url": "https://github.com/smellslikeml/experimental-vqasynth/blob/main/assets/warehouse_sample_1.jpeg?raw=true",
        "question": "Does the red forklift in the warehouse appear on the left side of the brown cardboard boxes that are stacked up?",
    }

    print("\n--- Running Benchmark ---")
    print(f"Image: {test_case['image_url']}")
    print(f"Question: {test_case['question']}")
    print(f"LessIsMore Enabled: {args.enable_less_is_more}")
    if args.enable_less_is_more:
        print(f"Budget: {args.budget}, Recent Ratio: {args.recent_ratio}")

    response, latency, num_tokens = run_benchmark(
        model, processor, test_case["image_url"], test_case["question"]
    )

    print("\n--- Results ---")
    print(f"Response: {response}")
    print(f"Total Latency: {latency:.4f} seconds")
    print(f"Generated Tokens: {num_tokens}")
    if num_tokens > 0:
        print(f"Per-Token Latency: {latency / num_tokens:.4f} seconds/token")


if __name__ == "__main__":
    main()
