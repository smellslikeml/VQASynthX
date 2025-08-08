import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5VLForConditionalGeneration
import io

def create_image(objects):
    """Creates a simple 256x256 image with colored shapes."""
    img = Image.new('RGB', (256, 256), color = 'white')
    draw = ImageDraw.Draw(img)
    for obj_color, obj_shape, bbox in objects:
        if obj_shape == 'cube':
            draw.rectangle(bbox, fill=obj_color)
        elif obj_shape == 'sphere':
            draw.ellipse(bbox, fill=obj_color)
    return img

def run_temporal_evaluation():
    """
    Evaluates a VQASynth-trained model on a simple temporal reasoning task
    inspired by LTLZinc's sequence-based benchmarks.
    """
    print("Loading model and processor...")
    # Use a VQASynth-trained model
    model_id = "remyxai/SpaceThinker-Qwen2.5VL-3B"
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = Qwen2_5VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Failed to load model. Ensure you are logged in via 'huggingface-cli login'. Error: {e}")
        return

    print("Model loaded.")

    # --- Create a minimal, self-contained dataset ---
    # Image 1: Red cube is on the left of the blue sphere
    image_before = create_image([
        ('red', 'cube', (50, 100, 100, 150)),
        ('blue', 'sphere', (150, 100, 200, 150))
    ])

    # Image 2: Red cube is on the right of the blue sphere
    image_after = create_image([
        ('blue', 'sphere', (50, 100, 100, 150)),
        ('red', 'cube', (150, 100, 200, 150))
    ])

    # Our temporal evaluation dataset
    eval_dataset = [
        {
            "images": [image_before, image_after],
            "question": "In the provided sequence of two images, describe the change in the position of the red cube relative to the blue sphere.",
            "expected_keywords": ["left", "right", "moved", "swapped"],
        },
        {
            "images": [image_before, image_after],
            "question": "Is the red cube on the right side of the blue sphere in the second image?",
            "expected_keywords": ["yes", "correct", "is on the right"],
        },
        {
            "images": [image_after, image_before],
            "question": "Considering the two images as a sequence, what happened to the red cube?",
            "expected_keywords": ["right", "left", "moved"],
        }
    ]

    print(f"\n--- Starting Evaluation on {len(eval_dataset)} samples ---")
    correct_predictions = 0

    for i, sample in enumerate(eval_dataset):
        prompt = processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": f"<|image_1|>\n<|image_2|>\n{sample['question']}"}],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(text=prompt, images=sample["images"], return_tensors="pt").to(model.device, torch.bfloat16)

        # Generate response
        generated_ids = model.generate(**inputs, max_new_tokens=100)
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # Clean up response to get just the assistant's answer
        answer_part = response.split("Assistant:")[-1].strip()

        print(f"\n--- Sample {i+1} ---")
        print(f"Question: {sample['question']}")
        print(f"Model Answer: {answer_part}")

        # Simple keyword-based evaluation
        is_correct = any(keyword.lower() in answer_part.lower() for keyword in sample['expected_keywords'])
        if is_correct:
            correct_predictions += 1
            print("Evaluation: CORRECT")
        else:
            print("Evaluation: INCORRECT")
        print(f"Expected keywords: {sample['expected_keywords']}")

    accuracy = (correct_predictions / len(eval_dataset)) * 100
    print(f"\n--- Evaluation Complete ---")
    print(f"Final Accuracy: {accuracy:.2f}% ({correct_predictions}/{len(eval_dataset)})")

if __name__ == "__main__":
    run_temporal_evaluation()
