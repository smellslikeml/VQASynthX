import torch
from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration

# This script demonstrates generating VQA samples based on conflicting stakeholder perspectives,
# an idea inspired by the simulation of different actors in the HoPeS project.

def setup_model():
    """Loads the LLaVA model and processor."""
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    # Sending to device before creating processor
    model.to("cuda:0")

    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    return model, processor

def generate_description(model, processor, image, persona_prompt):
    """Generates a description for an image from a specific persona's point of view."""
    prompt = f"USER: <image>\n{persona_prompt}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda:0", torch.float16)
    generate_ids = model.generate(**inputs, max_new_tokens=150)
    output_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    # Extract only the assistant's response
    assistant_response = output_text.split("ASSISTANT:")[-1].strip()
    return assistant_response

def main():
    """Main function to run the stakeholder simulation experiment."""
    print("Initializing LLaVA model... (This may take a moment and requires a GPU)")
    model, processor = setup_model()

    # --- 1. Setup Scene and Stakeholders ---
    # Using a sample image from VQASynth's assets.
    # A scene with both natural and man-made elements is ideal for this experiment.
    image_url = "https://github.com/smellslikeml/experimental-vqasynth/blob/main/assets/warehouse_sample_2.jpeg?raw=true"
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    
    # Personas inspired by the stakeholder types in HoPeS (e.g., Land Use vs. Environmental)
    persona_developer = "You are a real estate developer. Describe this scene, focusing on the opportunities for commercial development, logistics, and infrastructure."
    persona_environmentalist = "You are an environmentalist. Describe this scene, focusing on the environmental impact, the state of natural elements, and potential for ecological restoration."

    print("\n--- Generating Stakeholder Perspectives ---")

    # --- 2. Generate Descriptions from each Perspective ---
    print("\n[Developer Perspective]")
    desc_developer = generate_description(model, processor, image, persona_developer)
    print(desc_developer)

    print("\n[Environmentalist Perspective]")
    desc_environmentalist = generate_description(model, processor, image, persona_environmentalist)
    print(desc_environmentalist)

    # --- 3. Formulate a Synthesis Question ---
    # This question requires the model to understand the conflict implied by the two descriptions.
    synthesis_question = (
        "Given the two following perspectives on the provided image, what is the primary point of conflict or tension? Assume both perspectives are valid.\n\n"
        f'Perspective A (Developer): \"{desc_developer}\"\n'
        f'Perspective B (Environmentalist): \"{desc_environmentalist}\"'
    )

    # For this minimal example, we define an "ideal" answer conceptually.
    # A full pipeline would generate this, perhaps with a more powerful model like GPT-4.
    ideal_answer = (
        "The primary conflict is between the commercial use and potential expansion of the warehouse/industrial space "
        "and the preservation or restoration of the surrounding natural environment, including the trees and open land. "
        "The developer sees economic potential while the environmentalist sees ecological impact."
    )

    print("\n--- Generated VQA Sample for Multi-Perspective Reasoning ---")
    print(f"\n[Image]: {image_url}")
    print(f"\n[Synthesis Question]: {synthesis_question}")
    print(f"\n[Ideal Answer]: {ideal_answer}")
    print("\n-----------------------------------------------------------")


if __name__ == "__main__":
    main()
