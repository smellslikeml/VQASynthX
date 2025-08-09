import argparse
import os
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import numpy as np

# NOTE: The ReX library API is inferred from the source repository's structure.
# The exact class names and arguments might need minor adjustments.
from rex_xai.explanation.rex import CausalExplainer
from rex_xai.input.config import CausalArgs
from rex_xai.input.input_data import InputData


def get_vlm_classifier_fn(model, processor, question, target_answer_token_id):
    """
    Creates a classifier function that ReX can use. 
    This function takes a batch of perturbed images (numpy arrays) and returns the VLM's 
    confidence score (logit) for the first token of a target answer.
    """
    def classifier_fn(image_array: np.ndarray) -> np.ndarray:
        batch_size = image_array.shape[0]
        scores = np.zeros(batch_size)

        for i in range(batch_size):
            pil_image = Image.fromarray(image_array[i].astype(np.uint8))
            prompt = f"USER: <image>\n{question}\nASSISTANT:"
            
            inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to("cuda")

            with torch.no_grad():
                # Get logits for the very next token to be generated
                logits = model(**inputs).logits[:, -1, :]
                # Extract the score for our target token
                score = logits[0, target_answer_token_id].cpu().numpy()
                scores[i] = score
                
        # ReX expects a 2D array of (batch_size, num_classes)
        return scores.reshape(-1, 1)

    return classifier_fn


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading VLM model (Llava-1.5-7b)... This may take a moment.")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        quantization_config=quantization_config,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    print("Model loaded.")

    # We measure the model's confidence by looking at the logit for the first token of the target answer.
    target_answer_tokens = processor.tokenizer.encode(args.target_answer, add_special_tokens=False)
    if not target_answer_tokens:
        raise ValueError(f"Target answer '{args.target_answer}' could not be tokenized.")
    target_answer_token_id = target_answer_tokens[0]
    print(f"Targeting first token of '{args.target_answer}', Token ID: {target_answer_token_id}")

    image = Image.open(args.image_path).convert("RGB")
    image_np = np.array(image)

    vlm_classifier = get_vlm_classifier_fn(model, processor, args.question, target_answer_token_id)

    print("Setting up ReX explainer...")
    # Configure ReX to find a single, causal explanation
    rex_args = CausalArgs(
        method="cem",
        num_prototypes=1,
        num_mutants=150, # More mutants can lead to a clearer explanation
        coverage=0.9,
        occlusion_size=16 # Size of patches to block out
    )
    
    input_data = InputData(image_np)

    explainer = CausalExplainer(
        input_data=input_data,
        classifier_fn=vlm_classifier,
        causal_args=rex_args,
        target_class=0 # Our classifier has a single output
    )
    
    print("Running ReX to find the causal explanation...")
    explanation = explainer.explain()
    print("Explanation generated.")

    output_path = os.path.join(args.output_dir, "causal_explanation.png")
    explanation.visualise(save_path=output_path)
    print(f"Causal explanation saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate causal explanations for VLM spatial reasoning using ReX.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--question", type=str, required=True, help="The spatial question to ask the VLM.")
    parser.add_argument("--target_answer", type=str, default="Yes", help="The expected first word of the VLM's answer (e.g., 'Yes', 'The').")
    parser.add_argument("--output_dir", type=str, default="output/rex_explainer", help="Directory to save explanation artifacts.")
    
    # Example command using an image from the VQASynth repo:
    # python experiments/rex-causal-vqa-explainer/run.py \
    #   --image_path assets/warehouse_sample_1.jpeg \
    #   --question "Is the red forklift on the left?" \
    #   --target_answer "Yes"

    args = parser.parse_args()
    main(args)
