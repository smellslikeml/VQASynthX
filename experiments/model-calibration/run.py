import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Inspired by calibration_tester.py from the SOURCE repo, this script demonstrates
# temperature scaling to improve model calibration. In VQASynth, this could be
# applied to models that filter or classify generated data to ensure that their
# confidence scores are reliable, improving the quality of the final VQA dataset.


class SimpleClassifier(nn.Module):
    """A simple model to generate logits for our calibration demo."""

    def __init__(self, input_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, 50)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(50, num_classes)

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))


def generate_dummy_data(n_samples=1000, input_size=10, num_classes=2):
    """Generates synthetic logits and labels.
    We'll intentionally create a slightly miscalibrated model by training briefly.
    """
    model = SimpleClassifier(input_size, num_classes)
    # Intentionally make the model overconfident by using a simple dataset
    features = torch.randn(n_samples, input_size)
    # Create labels that are mostly correct but have some noise
    true_weights = torch.randn(input_size, num_classes)
    true_bias = torch.randn(num_classes)
    logits = features @ true_weights + true_bias
    labels = torch.argmax(logits, dim=1)
    # Add some label noise to prevent perfect separation and encourage overconfidence
    noise_mask = torch.rand(n_samples) < 0.15
    labels[noise_mask] = 1 - labels[noise_mask]

    # Briefly train the model to get some reasonable but not perfectly calibrated logits
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(20):
        optimizer.zero_grad()
        out_logits = model(features)
        loss = loss_fn(out_logits, labels)
        loss.backward()
        optimizer.step()

    # Get the final logits on the "validation" data (the same data here for simplicity)
    with torch.no_grad():
        final_logits = model(features)

    return final_logits, labels


class ModelWithTemperature(nn.Module):
    """
    A thin wrapper for a model to apply temperature scaling.
    This is the core idea from SOURCE's calibration_tester.py.
    """

    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        logits = self.model(x)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        # The key operation: dividing logits by the temperature.
        return logits / self.temperature


def find_optimal_temperature(logits, labels):
    """
    Find the optimal temperature T by minimizing NLL on a validation set.
    """
    nll_criterion = nn.CrossEntropyLoss()
    # Initial temperature
    temperature = nn.Parameter(torch.ones(1) * 1.0)
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def eval():
        optimizer.zero_grad()
        loss = nll_criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(eval)
    return temperature.item()


def calculate_ece(probs, labels, n_bins=15):
    """
    Calculates the Expected Calibration Error of a model.
    """
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()


def plot_reliability_diagram(uncal_probs, cal_probs, labels, n_bins=15):
    """Plots a reliability diagram to visualize calibration."""
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

    def get_bin_accuracies(probs):
        accuracies = []
        confidences = []
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (probs.max(1)[0] > bin_lower) & (probs.max(1)[0] <= bin_upper)
            if in_bin.sum() > 0:
                bin_acc = (probs.argmax(1)[in_bin] == labels[in_bin]).float().mean()
                bin_conf = probs.max(1)[0][in_bin].mean()
                accuracies.append(bin_acc.item())
                confidences.append(bin_conf.item())
            else:
                accuracies.append(0)
                confidences.append(0)  # or nan
        return accuracies, confidences

    uncal_accs, _ = get_bin_accuracies(uncal_probs)
    cal_accs, _ = get_bin_accuracies(cal_probs)

    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.plot(
        bin_centers,
        uncal_accs,
        marker="o",
        linestyle="-",
        color="blue",
        label="Uncalibrated Model",
    )
    plt.plot(
        bin_centers,
        cal_accs,
        marker="s",
        linestyle="-",
        color="red",
        label="Calibrated Model (Temp. Scaling)",
    )
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(True)
    plt.savefig("reliability_diagram.png")
    print("Saved reliability diagram to reliability_diagram.png")


if __name__ == "__main__":
    print("Generating dummy data and an overconfident model...")
    val_logits, val_labels = generate_dummy_data(n_samples=2000)

    # Get uncalibrated probabilities and ECE
    uncalibrated_probs = torch.softmax(val_logits, dim=1)
    ece_before = calculate_ece(uncalibrated_probs, val_labels)
    print(f"ECE before calibration: {ece_before:.4f}")

    # Find optimal temperature
    print("Finding optimal temperature...")
    optimal_t = find_optimal_temperature(val_logits, val_labels)
    print(f"Optimal temperature found: {optimal_t:.3f}")

    # Apply temperature and get calibrated probabilities and ECE
    calibrated_logits = val_logits / optimal_t
    calibrated_probs = torch.softmax(calibrated_logits, dim=1)
    ece_after = calculate_ece(calibrated_probs, val_labels)
    print(f"ECE after calibration: {ece_after:.4f}")

    # Plotting results
    print("Generating reliability diagram...")
    plot_reliability_diagram(uncalibrated_probs, calibrated_probs, val_labels)
