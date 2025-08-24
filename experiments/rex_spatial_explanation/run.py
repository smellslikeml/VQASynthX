import torch
from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration
from rex_xai.lib import rex_image
import numpy as np
import os

# --- 1. Setup Model and Processor ---
# This experiment uses a pre-trained LLaVA model as the VLM to be explained.
# VQASynth's goal is to fine-tune such models, so using a base model
# is a representative starting point for explainability.
print("Loading LLaVA model...")
model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to("cuda:0")
processor = AutoProcessor.from_pretrained(model_id)
print("Model loaded.")

# --- 2. Prepare Input Image and Prompt ---
# Use a sample image relevant to the VQASynth's domain (e.g., industrial scenes).
# This image contains clear objects for a spatial reasoning task.
image_url = "https://github.com/smellslikeml/experimental-vqasynth/raw/main/assets/warehouse_sample_1.jpeg"
raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
output_dir = "rex_vqa_outputs"
os.makedirs(output_dir, exist_ok=True)
raw_image.save(os.path.join(output_dir, "original_image.jpg"))


# The core idea is to frame a VQA task as a classification task that ReX can understand.
# We ask a binary (Yes/No) spatial question.
question = "Does the red forklift in warehouse appear on the left side of the brown cardboard boxes stacked?"
prompt = f"USER: <image>\n{question} Answer with only 'Yes' or 'No'.\nASSISTANT:"

# Define the target class for ReX. We want to explain why the model answered "Yes".
target_class_str = "Yes"
target_class_id = processor.tokenizer.encode(
    target_class_str, add_special_tokens=False
)[0]


# --- 3. Create a Classifier Wrapper for ReX ---
# ReX needs a function that takes a batch of images (numpy arrays) and returns
# prediction scores (logits) for a target class. This wrapper adapts the LLaVA model
# to fit that interface.
def vlm_classifier_fn(images_np):
    """
    Wrapper function to make the VLM compatible with ReX's classifier interface.

    Args:
        images_np (np.ndarray): A batch of images of shape (N, H, W, C).

    Returns:
        np.ndarray: A batch of logits of shape (N, vocab_size).
    """
    images_pil = [Image.fromarray(img) for img in images_np]

    # Preprocess inputs for the LLaVA model
    inputs = processor(
        text=[prompt] * len(images_pil), images=images_pil, return_tensors="pt"
    ).to("cuda:0", torch.float16)

    # Get the model's logits
    with torch.no_grad():
        outputs = model(**inputs)
        # We only care about the logit for the *first* token of the answer.
        logits = outputs.logits[:, -1, :]

    return logits.cpu().numpy()


# --- 4. Run ReX to get the Causal Explanation ---
# Instantiate the ReX explainer with the image and the VLM wrapper.
print("Running ReX to generate causal explanation...")
explanation = rex_image(
    image=np.array(raw_image),
    classifier_fn=vlm_classifier_fn,
    target_class=target_class_id,  # Explain the logit for the token "Yes"
    saliency_method="IntegratedGradients",  # A common choice for saliency
    regions_method="slic",  # How to segment the image
    n_regions=100,  # Number of segments to create
    n_samples=1000,  # Number of perturbations for responsibility calculation
    device="cuda:0",
)
print("ReX run complete.")

# --- 5. Save the Results ---
# The explanation object contains the responsibility map and the minimal explanation.
# The responsibility map shows the causal importance of each image region.
# The minimal explanation is an image containing only the most important regions.
print(f"Saving results to '{output_dir}' directory...")
explanation.save_responsibility_map(os.path.join(output_dir, "responsibility_map.png"))
explanation.save_minimal_explanation(
    os.path.join(output_dir, "minimal_explanation.png")
)

# Also save the explanation data for further analysis
explanation.save(os.path.join(output_dir, "explanation_data.dill"))

print("Experiment finished successfully.")
print(f"Original model prediction for the question '{question}':")

# Verify the original prediction
inputs = processor(text=prompt, images=raw_image, return_tensors="pt").to(
    "cuda:0", torch.float16
)
generate_ids = model.generate(**inputs, max_new_tokens=10)
full_answer = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
# Extract just the generated part
answer_only = full_answer.split("ASSISTANT:")[1].strip()
print(f"-> {answer_only}")
