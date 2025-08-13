import torch
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer


def setup_pipeline():
    """Initializes and returns the GLiClass pipeline."""
    # Use a small, public model for this experiment
    model_name = "knowledgator/gliclass-small-v1.0"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = GLiClassModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pipeline = ZeroShotClassificationPipeline(
        model, tokenizer, classification_type="multi-label", device=device
    )
    return pipeline


def get_sample_vqa_data():
    """Provides sample VQA data mimicking the output of a VQASynth stage."""
    return [
        {"id": 1, "question": "How far is the red car from the stop sign in feet?"},
        {
            "id": 2,
            "question": "Is the person wearing a blue shirt standing to the left of the green bench?",
        },
        {"id": 3, "question": "What color is the sofa in the living room?"},
        {
            "id": 4,
            "question": "Provide a detailed description of the objects on the desk.",
        },
        {
            "id": 5,
            "question": "Estimate the distance between the two boats on the water.",
        },
        {
            "id": 6,
            "question": "the from to is where?",
        },  # Example of a low-quality/malformed question
    ]


def main():
    """
    Main function to run the GLiClass filtering experiment.
    This script demonstrates how GLiClass can be used for zero-shot classification
    to categorize and filter synthetic VQA data, a potential step in the VQASynth pipeline.
    """
    print("Initializing GLiClass pipeline...")
    pipeline = setup_pipeline()

    print("\nLoading sample VQA data...")
    vqa_data = get_sample_vqa_data()

    # Define labels to categorize the VQA questions
    # These labels are chosen to reflect common spatial reasoning tasks
    labels = [
        "distance estimation",
        "relative positioning",
        "object attribute",
        "scene description",
        "malformed or low-quality",
    ]
    print(f"\nUsing classification labels: {labels}")

    print("\n--- Running VQA Classification ---")
    for item in vqa_data:
        question = item["question"]
        print(f"\nProcessing Question: '{question}'")

        # Classify the question using GLiClass
        results = pipeline(question, labels, threshold=0.3)[0]

        if not results:
            print("  -> No labels met the threshold.")
        else:
            print("  -> Predicted Labels:")
            for result in results:
                print(f"    - {result['label']}: {result['score']:.4f}")

    print("\n--- Experiment Finished ---")


if __name__ == "__main__":
    main()
