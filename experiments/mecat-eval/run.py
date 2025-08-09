import json
from collections import defaultdict
import os

def evaluate_spatial_vqa(data):
    """
    Evaluates VQA predictions based on predefined spatial reasoning categories,
    inspired by the MECAT benchmark's hierarchical evaluation structure.

    MECAT provides a fine-grained breakdown of audio tasks (e.g., by content
    type and complexity). We apply a similar philosophy to spatial reasoning
    in visual question answering, categorizing questions to get a nuanced
    view of model performance beyond a single accuracy score.

    Args:
        data (list): A list of dictionaries, where each dictionary contains
                     a 'ground_truth', 'prediction', and 'category' string.
    """
    category_scores = defaultdict(lambda: {'correct': 0, 'total': 0})

    for item in data:
        category = item.get('category')
        if not category:
            continue

        # Simple exact match for evaluation
        is_correct = str(item['ground_truth']).lower() == str(item['prediction']).lower()

        # Update scores for the specific category
        category_scores[category]['total'] += 1
        if is_correct:
            category_scores[category]['correct'] += 1

    # Calculate and print results
    print("--- MECAT-Inspired Fine-Grained VQA Evaluation ---")
    
    # Sort categories for consistent output
    sorted_categories = sorted(category_scores.keys())

    # Get max category length for pretty printing
    max_len = max(len(cat) for cat in sorted_categories) if sorted_categories else 0

    overall_correct = 0
    overall_total = 0

    for category in sorted_categories:
        stats = category_scores[category]
        total = stats['total']
        correct = stats['correct']
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"{category:<{max_len}} | Accuracy: {accuracy:6.2f}% ({correct}/{total})")
        overall_correct += correct
        overall_total += total
    
    print("-" * (max_len + 28))
    overall_accuracy = (overall_correct / overall_total * 100) if overall_total > 0 else 0
    print(f"{'Overall':<{max_len}} | Accuracy: {overall_accuracy:6.2f}% ({overall_correct}/{overall_total})")
    print("\nThis categorical breakdown provides deeper insight into model strengths and weaknesses,")
    print("mirroring the fine-grained evaluation approach of the MECAT benchmark.")


def get_sample_data():
    """
    Provides sample data representing model outputs for evaluation.
    In a real scenario, this would load a results file.
    The categories are inspired by MECAT's hierarchical task structure.
    - Type: Metric, Positional, Orientation
    - Subtask: Distance, Relative, Absolute
    - Complexity: Simple, Complex, Direct, Comparative
    """
    return [
      {
        "id": 1,
        "question": "How far is the red car from the blue truck in meters?",
        "ground_truth": "5.2",
        "prediction": "5.1",
        "category": "Metric/Distance/Direct"
      },
      {
        "id": 2,
        "question": "Is the red car closer to the blue truck or the building?",
        "ground_truth": "blue truck",
        "prediction": "blue truck",
        "category": "Metric/Distance/Comparative"
      },
      {
        "id": 3,
        "question": "Is the book to the left of the lamp?",
        "ground_truth": "yes",
        "prediction": "no",
        "category": "Positional/Relative/Simple"
      },
      {
        "id": 4,
        "question": "From the camera's viewpoint, what is behind the lamp?",
        "ground_truth": "the wall",
        "prediction": "the wall",
        "category": "Positional/Relative/Complex"
      },
      {
        "id": 5,
        "question": "Is the bottle standing upright?",
        "ground_truth": "yes",
        "prediction": "yes",
        "category": "Orientation/Absolute/Simple"
      },
      {
        "id": 6,
        "question": "Is the red car closer to the blue truck or the building?",
        "ground_truth": "blue truck",
        "prediction": "building",
        "category": "Metric/Distance/Comparative"
      }
    ]

def main():
    """
    Main function to load data and run the evaluation.
    This experiment demonstrates a new evaluation strategy.
    """
    print("Running MECAT-inspired Categorical Evaluation for Spatial VQA.\n")
    sample_data = get_sample_data()
    evaluate_spatial_vqa(sample_data)

if __name__ == "__main__":
    main()
