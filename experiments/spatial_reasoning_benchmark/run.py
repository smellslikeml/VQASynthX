import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re

# Inspired by the methodology in trustsuperset/privacy_llm_benchmark,
# this script establishes a baseline for spatial reasoning using a multiple-choice quiz.

SPATIAL_REASONING_QUESTIONS = [
    {
        "question": "A car is facing North. It turns 90 degrees to the right, then drives forward, then turns 180 degrees. What direction is it facing now?",
        "options": {"A": "North", "B": "South", "C": "East", "D": "West"},
        "answer": "D",
    },
    {
        "question": "You are in a room. The door is to your North and the window is to your West. If you turn to face the window and then turn 90 degrees left, what are you facing?",
        "options": {"A": "The door", "B": "The window", "C": "South", "D": "East"},
        "answer": "C",
    },
    {
        "question": "Object A is 5 meters to the east of Object B. Object C is 10 meters to the west of Object A. What is the position of Object C relative to Object B?",
        "options": {
            "A": "5 meters west",
            "B": "5 meters east",
            "C": "15 meters west",
            "D": "15 meters east",
        },
        "answer": "A",
    },
    {
        "question": "A box is placed on a table. A book is placed on top of the box. A pen is placed to the right of the box. Which object is highest?",
        "options": {"A": "The box", "B": "The table", "C": "The book", "D": "The pen"},
        "answer": "C",
    },
    {
        "question": "If you walk 3 blocks North, then 4 blocks East, what is the straight-line distance from your starting point in blocks?",
        "options": {"A": "3", "B": "4", "C": "5", "D": "7"},
        "answer": "C",
    },
    {
        "question": "A person is standing in the center of a circular room. The entrance is at the northernmost point. The exit is at the southernmost point. The person is facing east. Where is the exit relative to the person's field of view?",
        "options": {
            "A": "Directly behind them",
            "B": "To their right and slightly behind",
            "C": "To their left",
            "D": "Directly in front of them",
        },
        "answer": "B",
    },
]


def run_benchmark(model_id, questions):
    """Loads a model and evaluates it on the spatial reasoning quiz."""
    print(f"Starting benchmark for model: {model_id}")

    quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quantization_config, device_map="auto"
    )

    correct_answers = 0
    total_questions = len(questions)

    for i, item in enumerate(questions):
        print(f"\nProcessing question {i+1}/{total_questions}...")
        question_text = item["question"]
        options_text = "\n".join(
            [f"{key}: {value}" for key, value in item["options"].items()]
        )
        prompt = f"Question: {question_text}\n\nOptions:\n{options_text}\n\nChoose the correct option (A, B, C, or D). Answer with only the letter.\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate a short response
        outputs = model.generate(
            **inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id
        )
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the letter from the response
        # Assumes the answer is in the format "Answer: A"
        answer_part = response_text.split("Answer:")[-1].strip()

        # Use regex to find the first letter A, B, C, or D
        match = re.search(r"[A-D]", answer_part.upper())
        model_answer = match.group(0) if match else "None"

        correct_answer = item["answer"]
        print(f"Model's Answer: {model_answer}, Correct Answer: {correct_answer}")

        if model_answer == correct_answer:
            correct_answers += 1

    accuracy = (correct_answers / total_questions) * 100
    print(f"\n--- Benchmark Complete ---")
    print(f"Model: {model_id}")
    print(f"Total Questions: {total_questions}")
    print(f"Correct Answers: {correct_answers}")
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    # Using a strong, small, instruction-tuned model as a baseline.
    # Mistral-7B-Instruct is a good starting point.
    baseline_model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    run_benchmark(baseline_model_id, SPATIAL_REASONING_QUESTIONS)
