import numpy as np
import matplotlib.pyplot as plt

# Inspired by calibration_tester.py from the SOURCE repository.
# This script provides a tool to analyze the calibration of a model's
# confidence scores on a classification task, such as a binary VQA task
# ("Is object A to the left of object B?").

def get_bin_index(probability, number_of_bins):
    """
    Calculates the bin index for a given probability.
    
    Args:
        probability (float): The probability score, between 0 and 1.
        number_of_bins (int): The total number of bins.
    
    Returns:
        int: The index of the bin.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Probability must be between 0 and 1.")
    
    # Handle the edge case where probability is exactly 1.0
    if probability == 1.0:
        return number_of_bins - 1
        
    return int(probability * number_of_bins)

def generate_mock_vqa_outputs(num_samples=1000):
    """
    Generates synthetic model predictions and ground truth labels.
    This function simulates a slightly overconfident, but generally decent, model.
    
    Args:
        num_samples (int): The number of samples to generate.
    
    Returns:
        tuple: A tuple containing:
            - np.ndarray: An array of predicted probabilities for the positive class.
            - np.ndarray: An array of ground truth labels (0 or 1).
    """
    # Generate true underlying probabilities
    true_probs = np.random.rand(num_samples)
    
    # Simulate an overconfident model: it pushes probabilities towards 0 and 1
    # A power function is a simple way to simulate overconfidence.
    predicted_probs = np.clip(true_probs + 0.15 * np.sin(true_probs * 2 * np.pi), 0, 1)

    # Generate ground truth labels based on the true probabilities
    ground_truth = (np.random.rand(num_samples) < true_probs).astype(int)
    
    return predicted_probs, ground_truth

def plot_reliability_diagram(accuracies, confidences, number_of_bins):
    """
    Plots the reliability diagram.
    
    Args:
        accuracies (np.ndarray): The accuracy for each bin.
        confidences (np.ndarray): The average confidence for each bin.
        number_of_bins (int): The total number of bins.
    """
    # Filter out bins that had no samples and would result in NaN
    valid_indices = ~np.isnan(confidences) & ~np.isnan(accuracies)

    plt.figure(figsize=(8, 8))
    # Plot the model's calibration curve
    plt.plot(confidences[valid_indices], accuracies[valid_indices], marker='o', linestyle='-', label='Model Calibration')
    
    # Plot the line of perfect calibration
    plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Perfect Calibration')
    
    plt.xlabel('Average Confidence in Bin')
    plt.ylabel('Accuracy in Bin')
    plt.title('Calibration Reliability Diagram')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    
    output_path = 'calibration_reliability_diagram.png'
    plt.savefig(output_path)
    print(f"\nReliability diagram saved to {output_path}")

def main():
    """
    Main function to run the calibration analysis.
    """
    number_of_bins = 10
    num_samples = 5000

    print(f"Generating {num_samples} mock VQA model outputs...")
    # In a real scenario, these would be replaced by actual model outputs on a VQA dataset
    predicted_probs, ground_truth_labels = generate_mock_vqa_outputs(num_samples)

    # Arrays to store metrics for each bin
    bin_accuracies = np.zeros(number_of_bins)
    bin_confidences = np.zeros(number_of_bins)
    bin_counts = np.zeros(number_of_bins, dtype=int)
    
    print("Analyzing model calibration...")
    for i in range(num_samples):
        prob = predicted_probs[i]
        label = ground_truth_labels[i]
        
        # Determine the prediction (0 or 1) and associated confidence
        prediction = 1 if prob > 0.5 else 0
        confidence = prob if prediction == 1 else 1 - prob
        
        # Determine if the prediction was correct
        is_correct = 1 if prediction == label else 0

        # Place the sample in the correct bin based on its confidence score
        bin_index = get_bin_index(confidence, number_of_bins)
        
        bin_counts[bin_index] += 1
        bin_accuracies[bin_index] += is_correct
        bin_confidences[bin_index] += confidence

    # Calculate final accuracies and confidences, avoiding division by zero for empty bins.
    non_zero_bins = bin_counts > 0
    final_accuracies = np.full(number_of_bins, np.nan)
    final_confidences = np.full(number_of_bins, np.nan)

    final_accuracies[non_zero_bins] = bin_accuracies[non_zero_bins] / bin_counts[non_zero_bins]
    final_confidences[non_zero_bins] = bin_confidences[non_zero_bins] / bin_counts[non_zero_bins]
    
    print("\nCalibration Analysis Results:")
    print("===========================")
    print(f"{'Bin Center':<15}{'Avg Confidence':<20}{'Accuracy':<15}{'Count':<10}")
    bin_centers = np.arange(0, 1, 1/number_of_bins) + (0.5 / number_of_bins)
    for i in range(number_of_bins):
        conf_str = f"{final_confidences[i]:.3f}" if not np.isnan(final_confidences[i]) else 'N/A'
        acc_str = f"{final_accuracies[i]:.3f}" if not np.isnan(final_accuracies[i]) else 'N/A'
        print(f"{bin_centers[i]:<15.2f}{conf_str:<20}{acc_str:<15}{bin_counts[i]:<10}")

    plot_reliability_diagram(final_accuracies, final_confidences, number_of_bins)

if __name__ == "__main__":
    # Ensure matplotlib is installed
    try:
        import matplotlib
    except ImportError:
        print("Matplotlib not found. Please install it using: pip install matplotlib")
        exit(1)
    main()
