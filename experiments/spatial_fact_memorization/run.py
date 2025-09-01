import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import random
import pandas as pd

# --- 1. Data Generation ---
# Inspired by the synthetic data generation in itl/apps/large_scale/data/HF_dataset_generation.py


def generate_spatial_facts(num_facts=50):
    """Generates a dataset of simple spatial relationship facts."""
    objects = [
        "red cube",
        "blue sphere",
        "green pyramid",
        "yellow cylinder",
        "purple cone",
    ]
    locations = ["the table", "the floor", "the shelf", "the box", "the chair"]
    relations = [
        "is to the left of",
        "is to the right of",
        "is on top of",
        "is underneath",
    ]

    facts = set()
    while len(facts) < num_facts:
        obj1 = random.choice(objects)
        obj2 = random.choice(objects)
        relation = random.choice(relations)

        if obj1 == obj2:
            loc = random.choice(locations)
            # Create a location-based fact instead
            fact = f"The {obj1} {relation} {loc}."
        else:
            fact = f"The {obj1} {relation} the {obj2}."
        facts.add(fact)

    return list(facts)


def create_dataset(facts):
    """Creates a Hugging Face Dataset from a list of facts."""
    prompts = []
    for fact in facts:
        # Simple text format for language modeling
        prompts.append({"text": f"Fact: {fact}"})
    return Dataset.from_list(prompts)


# --- 2. Model Training (In-Weight Learning) ---
# Inspired by the PEFT/LoRA fine-tuning in itl/apps/large_scale/training/finetune_parallelized.py


def train_model(
    dataset, model_name="EleutherAI/pythia-160m", output_dir="./spatial_facts_model"
):
    """Fine-tunes a model on the spatial facts dataset using LoRA."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use bfloat16 for better performance on modern GPUs
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=128
        )

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        logging_steps=10,
        save_strategy="no",
        learning_rate=2e-4,
        fp16=False,  # fp16 is deprecated for torch>2.0, bfloat16 is preferred
        bf16=(
            True
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else False
        ),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("\n--- Starting In-Weight Fine-Tuning (LoRA) ---")
    trainer.train()
    print("--- Fine-Tuning Complete ---")
    return model.merge_and_unload(), tokenizer


# --- 3. Evaluation ---


def evaluate_recall(model, tokenizer, test_facts):
    """Evaluates how well the model recalls the facts by completing them."""
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    correct = 0
    results = []
    for fact in test_facts:
        parts = fact.split()
        prompt_text = " ".join(parts[:4]) + " "
        expected_completion = " ".join(parts[4:])

        inputs = tokenizer(f"Fact: {prompt_text}", return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = generated_text.replace(f"Fact: {prompt_text}", "").strip()

        is_correct = expected_completion.lower() in completion.lower()
        if is_correct:
            correct += 1
        results.append(
            {
                "prompt": prompt_text,
                "expected": expected_completion,
                "generated": completion,
                "correct": is_correct,
            }
        )

    return correct / len(test_facts), pd.DataFrame(results)


def evaluate_in_context(model_name, test_facts):
    """Simulates a perfect retrieval tool by providing facts in-context."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    correct = 0
    results = []
    for fact in test_facts:
        parts = fact.split()
        question_prompt = " ".join(parts[:4]) + "?"
        expected_answer = " ".join(parts[4:])

        context = f"Using the following fact, answer the question.\nFact: {fact}\nQuestion: {question_prompt}\nAnswer:"
        inputs = tokenizer(context, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text.replace(context, "").strip()

        is_correct = expected_answer.lower() in answer.lower()
        if is_correct:
            correct += 1
        results.append(
            {
                "prompt": question_prompt,
                "expected": expected_answer,
                "generated": answer,
                "correct": is_correct,
            }
        )

    return correct / len(test_facts), pd.DataFrame(results)


def main():
    """Main function to run the experiment."""
    NUM_FACTS = 20
    BASE_MODEL = "EleutherAI/pythia-160m"

    print(f"--- Step 1: Generating {NUM_FACTS} synthetic spatial facts ---")
    facts = generate_spatial_facts(num_facts=NUM_FACTS)
    print("Sample fact:", facts[0])

    train_facts, test_facts = (
        facts[: int(NUM_FACTS * 0.8)],
        facts[int(NUM_FACTS * 0.8) :],
    )
    train_dataset = create_dataset(train_facts)

    print(f"\n--- Step 2: Fine-tuning {BASE_MODEL} on facts (In-Weight Learning) ---")
    fine_tuned_model, tokenizer = train_model(train_dataset, model_name=BASE_MODEL)

    print("\n--- Step 3: Evaluating model recall on held-out facts ---")

    ft_accuracy, ft_results = evaluate_recall(fine_tuned_model, tokenizer, test_facts)
    print("\n--- Evaluation Results: Fine-Tuned Model (In-Weight) ---")
    print(f"Recall Accuracy: {ft_accuracy:.2%}")
    print(ft_results.head())

    print("\n--- Evaluation Results: Base Model (In-Context/Tool-Use) ---")
    base_accuracy, base_results = evaluate_in_context(BASE_MODEL, test_facts)
    print(f"Recall Accuracy: {base_accuracy:.2%}")
    print(base_results.head())

    print("\n--- Experiment Summary ---")
    summary = pd.DataFrame(
        {
            "Method": [
                "In-Weight Learning (Fine-Tuned)",
                "In-Context Learning (RAG Simulation)",
            ],
            "Recall Accuracy": [f"{ft_accuracy:.2%}", f"{base_accuracy:.2%}"],
        }
    )
    print(summary.to_string(index=False))
    print(
        "\nThis experiment demonstrates the trade-off between memorizing facts in model weights"
    )
    print(
        "vs. retrieving them from an external source, a core concept from the 'In-Tool Learning' paper."
    )


if __name__ == "__main__":
    main()
