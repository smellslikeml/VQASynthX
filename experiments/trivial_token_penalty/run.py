import torch
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    LogitsProcessor,
    LogitsProcessorList,
)

# --- 1. Define the Custom LogitsProcessor inspired by PC-Sampler ---
# The PC-Sampler paper proposes a "Calibrated Confidence Score" to suppress
# the selection of trivial tokens by incorporating frequency-based adjustments.
# We adapt this idea to an autoregressive setting by creating a LogitsProcessor
# that directly penalizes a predefined list of high-frequency, low-information tokens.


class TrivialTokenPenalizer(LogitsProcessor):
    """
    A LogitsProcessor that penalizes a specific list of trivial token IDs.
    This encourages the model to generate more substantive, less conversational content,
    adapting the core idea of PC-Sampler's Calibrated Confidence Score for autoregressive models.

    Args:
        trivial_token_ids (list[int]): A list of token IDs to be penalized.
        penalty (float): The penalty to apply. A value > 1.0 reduces the probability, a large penalty effectively suppresses the token.
    """

    def __init__(self, trivial_token_ids: list[int], penalty: float = 100.0):
        if not isinstance(trivial_token_ids, list) or not all(
            isinstance(i, int) for i in trivial_token_ids
        ):
            raise ValueError("`trivial_token_ids` has to be a list of integers.")
        if not penalty > 0:
            raise ValueError(
                f"`penalty` has to be a strictly positive float, but is {penalty}"
            )

        self.trivial_token_ids = trivial_token_ids
        self.penalty = penalty

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # For each token in our trivial list, apply a penalty by dividing its logit score.
        # This makes it much less likely to be sampled.
        scores[:, self.trivial_token_ids] /= self.penalty
        return scores


# --- 2. Setup Model and Processor ---
def setup_model_and_processor(model_id="llava-hf/llava-1.5-7b-hf"):
    """Loads the VLM and its associated processor."""
    print(f"Loading model: {model_id}...")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda:0")
    processor = AutoProcessor.from_pretrained(model_id)
    print("Model loaded successfully.")
    return model, processor


def get_trivial_token_ids(processor):
    """
    Identifies token IDs for common, low-information words and punctuation.
    This is a heuristic list, analogous to the reference corpus in PC-Sampler.
    """
    trivial_tokens = [
        ".",
        ",",
        "!",
        "?",
        ";",
        ":",
        "-",
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "so",
        "well",
        "actually",
        "basically",
        "like",
        "just",
        "So",
        "Well",
        "Actually",
        "Basically",
        "Like",
        "Just",
    ]
    # Convert tokens to IDs, ignoring those not in the tokenizer's vocabulary
    token_ids = processor.tokenizer.convert_tokens_to_ids(trivial_tokens)
    return [
        tid
        for tid in token_ids
        if tid is not None and tid != processor.tokenizer.unk_token_id
    ]


# --- 3. Run Generation Experiment ---
def run_generation(model, processor, prompt, image=None, logits_processor=None):
    """Runs a single generation pass with or without a logits processor."""
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(
        "cuda:0", torch.float16
    )

    processor_list = (
        LogitsProcessorList([logits_processor]) if logits_processor else None
    )

    generate_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        logits_processor=processor_list,
        do_sample=False,  # Use greedy decoding for reproducibility
    )
    return processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


if __name__ == "__main__":
    # Note: This script does not require an image for this demonstration,
    # as we are focused on the text generation characteristics.
    # The prompt simulates the input to the CoT reasoning stage.
    model_id = (
        "llava-hf/llava-1.5-7b-hf"  # A standard VLM used in VQASynth-like pipelines
    )
    model, processor = setup_model_and_processor(model_id)

    # This prompt is designed to elicit a Chain-of-Thought response.
    prompt = "USER: <image>\nBased on the image, describe the spatial relationship between the red forklift and the cardboard boxes. Think step-by-step. ASSISTANT:"

    print("\n" + "=" * 50)
    print("Running BASELINE generation (without penalty)...")
    print("=" * 50)
    baseline_output = run_generation(model, processor, prompt)
    print(baseline_output)

    print("\n" + "=" * 50)
    print("Running generation with TRIVIAL TOKEN PENALTY...")
    print("=" * 50)
    # Instantiate our custom logits processor
    trivial_ids = get_trivial_token_ids(processor)
    print(f"Penalizing {len(trivial_ids)} trivial token IDs.")
    penalizer = TrivialTokenPenalizer(trivial_token_ids=trivial_ids, penalty=100.0)

    penalized_output = run_generation(
        model, processor, prompt, logits_processor=penalizer
    )
    print(penalized_output)

    print("\n" + "=" * 50)
    print("COMPARISON")
    print("=" * 50)
    print(f"Baseline Output:\n{baseline_output.split('ASSISTANT:')[1].strip()}")
    print("-" * 20)
    print(f"Penalized Output:\n{penalized_output.split('ASSISTANT:')[1].strip()}")
    print(
        "\nExperiment complete. Observe if the penalized output is more concise and less conversational."
    )
