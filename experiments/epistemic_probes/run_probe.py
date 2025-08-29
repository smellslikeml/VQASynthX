import torch
from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Inspired by the epistemic markers used in the MarPT study to probe for variance in VLM responses.
# SOURCE: HKUST-KnowComp/MarPT/marker/
EPISTEMIC_MARKERS = [
    "",  # Baseline - no marker
    "Possibly, ",
    "Perhaps, ",
    "Maybe, ",
    "It is likely that ",
    "It is plausible that ",
    "It seems that ",
    "I guess that ",
]

# We test multiple common VQA phrasings to see if the effect is consistent.
QUESTION_TEMPLATES = [
    "{marker}is the {subject} to the {relation} of the {object}?",
    "Confirm: {marker}the {subject} is to the {relation} of the {object}.",
    "Does it appear that {marker}the {subject} is to the {relation} of the {object}?",
]


def run_probe(model, processor, image, subject, relation, object):
    """
    Probes a VLM's response consistency to a spatial question phrased
    with different epistemic uncertainty markers.

    Args:
        model: The loaded VLM model.
        processor: The model's preprocessor.
        image (PIL.Image): The input image.
        subject (str): The primary object in the question.
        relation (str): The spatial relationship (e.g., 'left').
        object (str): The reference object.
    """
    print(f"--- Running Epistemic Probe ---")
    print(f"Image: A warehouse scene")
    print(f"Base Query: Is the '{subject}' to the '{relation}' of the '{object}'?")
    print("-" * 30)

    results = {}

    for template in QUESTION_TEMPLATES:
        template_key = template.split(" ")[0].replace(":", "")
        print(f"\nTesting Template: '{template}'\n")
        results[template_key] = []
        for marker in EPISTEMIC_MARKERS:
            question = template.format(
                marker=marker, subject=subject, relation=relation, object=object
            )

            prompt = f"USER: <image>\n{question}\nASSISTANT:"

            inputs = processor(text=prompt, images=image, return_tensors="pt").to(
                model.device
            )

            generate_ids = model.generate(**inputs, max_new_tokens=75)
            response_text = processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            assistant_response = response_text.split("ASSISTANT:")[1].strip()

            print(f"  Prompt: {question}")
            print(f"  Response: {assistant_response}\n")
            results[template_key].append(
                {"prompt": question, "response": assistant_response}
            )

    return results


def main():
    """
    Main function to load a model and run the probe with a sample case.
    """
    # Using a standard VLM that VQASynth might be used to fine-tune
    model_id = "llava-hf/llava-1.5-7b-hf"
    print(f"Loading model: {model_id}...")

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda")

    processor = AutoProcessor.from_pretrained(model_id)

    # Sample image from the VQASynth README
    image_url = "https://github.com/smellslikeml/experimental-vqasynth/blob/main/assets/warehouse_sample_1.jpeg?raw=true"
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except Exception as e:
        print(f"Failed to load image from URL: {e}")
        print("Please ensure you have an internet connection.")
        return

    # Deconstructing the question from the VQASynth README for the probe
    subject = "red forklift"
    relation = "left"
    object = "brown cardboard boxes"

    run_probe(model, processor, image, subject, relation, object)


if __name__ == "__main__":
    main()
