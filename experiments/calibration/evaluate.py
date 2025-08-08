import numpy as np
import matplotlib.pyplot as plt

# Core logic adapted from calibration_tester.py in the SOURCE repo.

def get_bin_index(probability, number_of_bins):
    """Calculates the bin index for a given probability."""
    if probability < 0.0 or probability > 1.0:
        # Handle potential floating point errors
        probability = np.clip(probability, 0.0, 1.0)
    answer = int(probability * number_of_bins)
    if answer == number_of_bins:
        answer -= 1
    return answer

def calculate_ece(confidences, accuracies, counts):
    """Calculate Expected Calibration Error (ECE)."""
    total_samples = np.sum(counts)
    if total_samples == 0:
        return 0
    ece = 0
    for i in range(len(confidences)):
        if counts[i] > 0:
            ece += (counts[i] / total_samples) * np.abs(accuracies[i] - confidences[i])
    return ece

def evaluate_calibration(predictions, labels, number_of_bins=10):
    """Evaluates calibration for a set of predictions and labels."""
    bin_accuracies = np.zeros(number_of_bins)
    bin_confidences = np.zeros(number_of_bins)
    bin_counts = np.zeros(number_of_bins)

    for i in range(len(predictions)):
        confidence = np.max(predictions[i])
        predicted_label = np.argmax(predictions[i])
        true_label = labels[i]
        
        is_correct = 1 if predicted_label == true_label else 0
        
        bin_idx = get_bin_index(confidence, number_of_bins)
        
        bin_counts[bin_idx] += 1
        bin_accuracies[bin_idx] += is_correct
        bin_confidences[bin_idx] += confidence

    # Avoid division by zero
    non_zero_bins = bin_counts > 0
    bin_accuracies[non_zero_bins] /= bin_counts[non_zero_bins]
    bin_confidences[non_zero_bins] /= bin_counts[non_zero_bins]
    
    ece = calculate_ece(bin_confidences, bin_accuracies, bin_counts)
    
    return ece, bin_accuracies, bin_confidences, bin_counts

def plot_reliability_diagram(ece, confidences, accuracies, number_of_bins, temperature):
    """Plots a reliability diagram."""
    bin_edges = np.linspace(0, 1, number_of_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    plt.figure(figsize=(6, 6))
    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    # Plot model's calibration
    plt.bar(bin_centers, accuracies, width=1.0/number_of_bins, edgecolor='black', label='Model Accuracy')
    
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'Reliability Diagram (T={temperature})\nECE = {ece:.4f}')
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig(f'reliability_diagram_T_{temperature}.png')
    print(f"Saved reliability diagram to reliability_diagram_T_{temperature}.png")

def main():
    """
    Main function to run the calibration experiment.
    This is a simulation. A real implementation would load a VLM and dataset.
    """
    print("Running VLM calibration evaluation experiment...")
    
    # --- SIMULATED DATA ---
    # In a real scenario, this would come from a VLM evaluated on a dataset
    # like 'remyxai/OpenSpaces_MC_R1'.
    # We simulate an overconfident model.
    num_samples = 5000
    num_classes = 4 # e.g., for Multiple Choice A, B, C, D
    # Generate logits for an overconfident model
    np.random.seed(42)
    logits = np.random.randn(num_samples, num_classes) * 2.5 
    # True labels
    true_labels = np.random.randint(0, num_classes, num_samples)
    # Make the model "correct" about 70% of the time, but with high confidence
    for i in range(num_samples):
        if np.random.rand() < 0.7:
            logits[i, true_labels[i]] += 5 # Boost the logit for the correct class
    # --- END SIMULATION ---
    
    number_of_bins = 15
    temperatures_to_test = [1.0, 1.5, 2.0, 2.5]

    print(f"Evaluating calibration for {len(temperatures_to_test)} temperatures...")
    
    for T in temperatures_to_test:
        # Apply temperature scaling to logits
        scaled_logits = logits / T
        
        # Get probabilities via softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Evaluate calibration
        ece, accuracies, confidences, _ = evaluate_calibration(probabilities, true_labels, number_of_bins)
        
        print(f"\n--- Temperature: {T} ---")
        print(f"Expected Calibration Error (ECE): {ece:.4f}")
        
        # Plot and save diagram
        plot_reliability_diagram(ece, confidences, accuracies, number_of_bins, T)

if __name__ == "__main__":
    main()
