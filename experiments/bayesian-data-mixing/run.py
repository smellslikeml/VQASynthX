import torch
import numpy as np
import os
import random
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf_discrete

# --- VQASynth Problem Definition ---
# These are the different types of spatial reasoning questions we can generate.
# Our goal is to find the best mixture of these to train a VLM.
QUESTION_DOMAINS = ["distance", "orientation", "existence", "comparison"]
N_DOMAINS = len(QUESTION_DOMAINS)

# This is our synthetic "ground truth" - the unknown optimal mixture we want to find.
# For this experiment, let's assume a mix heavy on distance and orientation is best.
TRUE_OPTIMAL_MIXTURE = torch.tensor([0.4, 0.4, 0.1, 0.1])


def evaluate_mixture(weights: torch.Tensor) -> torch.Tensor:
    """
    A synthetic black-box function that simulates training a VLM on a data
    mixture and evaluating its performance. The score is higher the closer
    the weights are to the TRUE_OPTIMAL_MIXTURE.
    This replaces the need for actual model training in this experiment.
    """
    # Ensure weights are a 2D tensor for batch processing
    if weights.ndim == 1:
        weights = weights.unsqueeze(0)

    # L1 distance from the optimal mixture, scaled to be a "performance score"
    # A lower distance means a higher score.
    distance = torch.linalg.vector_norm(
        weights - TRUE_OPTIMAL_MIXTURE.to(weights), ord=1, dim=-1
    )
    # Convert distance to a score between ~0 and 1, with noise.
    score = torch.exp(-5 * distance) + torch.randn(weights.shape[0]) * 0.02
    return score.unsqueeze(-1)


def generate_initial_data(n_points: int = 5):
    """Generates some initial random mixtures and their scores to seed the GP model."""
    # Using a Dirichlet distribution to generate random weights that sum to 1.
    train_x = torch.distributions.Dirichlet(torch.ones(N_DOMAINS)).sample((n_points,))
    train_y = evaluate_mixture(train_x)
    return train_x, train_y


# --- Bayesian Optimization Setup (adapted from ADMIRE-BayesOpt) ---


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def initialize_model(train_x, train_obj):
    """Initializes and fits a SingleTaskGP model."""
    model = SingleTaskGP(train_x, train_obj)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def optimize_acqf_and_get_observation(acq_func, choices):
    """Optimizes the acquisition function over a discrete set of choices."""
    candidate_weights, _ = optimize_acqf_discrete(
        acq_function=acq_func,
        q=1,
        choices=choices,
        unique=True,
    )
    # The output of optimize_acqf_discrete is the selected choice itself
    new_x = candidate_weights
    new_obj = evaluate_mixture(new_x)
    return new_x, new_obj


if __name__ == "__main__":
    seed_everything(42)
    tkwargs = {"dtype": torch.double, "device": "cpu"}  # No GPU needed for this sim

    # 1. Generate a discrete set of possible mixture candidates to choose from
    # This is analogous to the pre-defined mixture ratios in the source repo.
    N_CANDIDATES = 100
    candidate_choices = torch.distributions.Dirichlet(torch.ones(N_DOMAINS)).sample(
        (N_CANDIDATES,)
    )
    candidate_choices = candidate_choices.to(**tkwargs)

    # 2. Generate initial data to start the optimization
    N_INIT = 5
    train_X, train_Y = generate_initial_data(n_points=N_INIT)
    train_X, train_Y = train_X.to(**tkwargs), train_Y.to(**tkwargs)

    print(
        f"Running Bayesian Optimization to find the best data mixture for {N_DOMAINS} VQA domains."
    )
    print(f"Starting with {N_INIT} random samples.")

    # 3. Optimization loop
    N_ITERATIONS = 10
    for i in range(N_ITERATIONS):
        # Fit the Gaussian Process model
        model = initialize_model(train_X, train_Y)

        # Define the acquisition function (Expected Improvement)
        best_f = train_Y.max()
        acq_function = qExpectedImprovement(model=model, best_f=best_f)

        # Optimize acquisition function to find the next best point to evaluate
        new_x, new_y = optimize_acqf_and_get_observation(
            acq_function, candidate_choices
        )

        # Add the new data to our training set
        train_X = torch.cat([train_X, new_x])
        train_Y = torch.cat([train_Y, new_y])

        print(f"Iteration {i+1}/{N_ITERATIONS}:")
        print(
            f"  - Recommended mixture to test: {dict(zip(QUESTION_DOMAINS, new_x.squeeze().tolist()))}"
        )
        print(f"  - Observed performance (score): {new_y.item():.4f}")

    # 4. Recommend the best mixture found
    best_observed_idx = train_Y.argmax()
    best_mixture = train_X[best_observed_idx]
    best_score = train_Y[best_observed_idx]

    print("\n" + "=" * 50)
    print("Bayesian Optimization Complete.")
    print(
        f"Ground Truth Optimal Mixture: {dict(zip(QUESTION_DOMAINS, TRUE_OPTIMAL_MIXTURE.tolist()))}"
    )
    print(f"Best mixture found after {N_INIT + N_ITERATIONS} evaluations:")
    print(
        f"  - Mixture: {dict(zip(QUESTION_DOMAINS, [round(x, 3) for x in best_mixture.squeeze().tolist()]))}"
    )
    print(f"  - Score: {best_score.item():.4f}")
    print("=" * 50)
