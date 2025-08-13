import uuid
from pathlib import Path

# This logger is sourced from the causal-reasoning-in-pieces repository
# and added to the project's utils.
from experiments.utils.experiment_logger import ExperimentLogger


def generate_fake_vqa_data(num_samples: int):
    """Generates a list of fake VQA data records for demonstration."""
    records = []
    for i in range(num_samples):
        record = {
            "image_id": f"img_{i:04d}.jpg",
            "question": f"Is object A to the left of object B in image {i}?",
            "answer": "Yes" if i % 2 == 0 else "No",
            "reasoning_chain": "1. Identified Object A. 2. Identified Object B. 3. Determined relative position. 4. Formulated answer.",
            "confidence": 0.95 - (i * 0.01),
            "generation_model": "gpt-4-vision-preview",
        }
        records.append(record)
    return records


def main():
    """
    Demonstrates the use of the ExperimentLogger for tracking VQA data generation.
    """
    print("Starting VQA data generation logging demo...")

    # Define output directory for logs
    output_dir = Path("output/logging_demo")

    # Initialize the logger with a unique job ID
    job_id = f"vqa-gen-{uuid.uuid4().hex[:8]}"
    logger = ExperimentLogger(logs_dir=output_dir, job_id=job_id)

    print(f"Logger initialized. Log file will be saved in: {logger.log_file.parent}")

    # Generate and log a single record using append()
    print("Logging a single record...")
    single_record = generate_fake_vqa_data(1)[0]
    single_record["notes"] = (
        "Logged individually"  # Logger dynamically handles new columns
    )
    logger.append(single_record)

    # Generate and log a batch of records using append_many()
    print("Logging a batch of records...")
    batch_records = generate_fake_vqa_data(5)
    logger.append_many(batch_records)

    print("Demo finished successfully.")
    print(f"Please check the CSV log file: {logger.log_file}")


if __name__ == "__main__":
    main()
