import numpy as np
from scipy.optimize import minimize
import itertools

# This script is a minimal, self-contained proof-of-concept adapting the core ideas
# from DRIVE's `maximum-likelihood-constraint-inference/gridworld.py`.
# It demonstrates learning an implicit behavioral constraint (avoiding a region)
# from a set of expert demonstrations (trajectories).

# 1. Environment and Problem Setup
# A simple 5x5 grid world
GRID_SIZE = 5
# The constraint we want to learn: an implicit "danger" or "keep-out" zone
FORBIDDEN_REGION = [(2, 2), (2, 3), (3, 2), (3, 3)]
START_STATE = (0, 0)
GOAL_STATE = (4, 4)

# Actions: UP, DOWN, LEFT, RIGHT, STAY
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

# 2. Expert Demonstrations
# These are optimal paths that implicitly obey the constraint (avoid the forbidden region).
DEMONSTRATIONS = [
    [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (4, 4)],
    [(0, 0), (1, 0), (1, 1), (2, 1), (3, 1), (4, 1), (4, 2), (4, 3), (4, 4)],
]


# 3. Feature Engineering (from DRIVE/mdp.py)
# Define features for each state. The model will learn weights for these features.
def get_features(state):
    """Returns a feature vector for a given state."""
    # Feature 1: Is the state in the forbidden region?
    in_forbidden_region = 1.0 if state in FORBIDDEN_REGION else 0.0
    # Feature 2: A constant bias feature
    bias = 1.0
    return np.array([in_forbidden_region, bias])


NUM_FEATURES = get_features((0, 0)).shape[0]


# 4. MDP Solver - Value Iteration (from DRIVE/mdp.py)
def value_iteration(weights):
    """Performs value iteration to find the optimal policy for a given set of weights."""
    states = list(itertools.product(range(GRID_SIZE), range(GRID_SIZE)))
    V = {s: 0 for s in states}
    gamma = 0.9  # Discount factor
    epsilon = 1e-3  # Convergence threshold

    while True:
        delta = 0
        for s in states:
            if s == GOAL_STATE:
                continue
            v = V[s]

            q_values = []
            for action in ACTIONS:
                next_s = (s[0] + action[0], s[1] + action[1])
                # Check bounds
                if not (0 <= next_s[0] < GRID_SIZE and 0 <= next_s[1] < GRID_SIZE):
                    next_s = s  # Stay in place if action is invalid

                cost = np.dot(weights, get_features(s))
                q_values.append(cost + gamma * V.get(next_s, 0))

            V[s] = min(q_values)
            delta = max(delta, abs(v - V[s]))
        if delta < epsilon:
            break

    # Extract policy
    policy = {}
    for s in states:
        if s == GOAL_STATE:
            policy[s] = (0, 0)
            continue

        best_action = None
        min_q = float("inf")
        for action in ACTIONS:
            next_s = (s[0] + action[0], s[1] + action[1])
            if not (0 <= next_s[0] < GRID_SIZE and 0 <= next_s[1] < GRID_SIZE):
                next_s = s
            cost = np.dot(weights, get_features(s))
            q_val = cost + gamma * V.get(next_s, 0)
            if q_val < min_q:
                min_q = q_val
                best_action = action
        policy[s] = best_action

    return policy


# 5. Maximum Likelihood Constraint Inference (from DRIVE/gridworld.py)
def log_likelihood(weights, demonstrations, policy_cache):
    """Calculates the negative log-likelihood of demonstrations given weights."""
    weights_tuple = tuple(weights)
    if weights_tuple in policy_cache:
        policy = policy_cache[weights_tuple]
    else:
        policy = value_iteration(weights)
        policy_cache[weights_tuple] = policy

    ll = 0.0
    for demo in demonstrations:
        for i in range(len(demo) - 1):
            state = demo[i]
            # The demonstrated action is the vector from current to next state
            action = (demo[i + 1][0] - state[0], demo[i + 1][1] - state[1])

            # If the demonstrated action matches the optimal policy, likelihood is high.
            if policy.get(state) == action:
                ll += 0  # log(1) = 0
            else:
                # If it doesn't match, penalize heavily.
                # A proper implementation would use softmax over Q-values.
                # For this PoC, a large penalty suffices to guide optimization.
                ll -= 10

    return -ll  # We minimize the negative log-likelihood


def run_inference():
    """Main function to run the constraint inference experiment."""
    print("Starting Behavioral Constraint Inference Experiment...")
    print(
        f"Goal: Learn weights to explain why demonstrators avoid the region: {FORBIDDEN_REGION}"
    )
    print("\n--- Demonstrations ---")
    for i, demo in enumerate(DEMONSTRATIONS):
        print(f"Demo {i+1}: {demo}")

    # Initial guess for weights
    initial_weights = np.random.rand(NUM_FEATURES)
    # Normalize to prevent scale issues
    initial_weights /= np.linalg.norm(initial_weights)

    print(f"\nInitial Weights (random): {initial_weights}")

    # Cache to store computed policies and speed up optimization
    policy_cache = {}

    # Run optimization
    result = minimize(
        log_likelihood,
        initial_weights,
        args=(DEMONSTRATIONS, policy_cache),
        method="SLSQP",
        bounds=[(-10, 10)] * NUM_FEATURES,  # Constrain weights to a reasonable range
    )

    inferred_weights = result.x
    # Normalize for interpretability
    inferred_weights /= np.linalg.norm(inferred_weights)

    print("\n--- Inference Results ---")
    print(f"Optimization successful: {result.success}")
    print(f"Final Inferred Weights (Normalized): {inferred_weights}")

    print("\n--- Interpretation ---")
    feature_names = ["In Forbidden Region", "Bias"]
    for name, weight in zip(feature_names, inferred_weights):
        print(f"  - Weight for '{name}': {weight:.4f}")

    if (
        inferred_weights[0] > 0.5
    ):  # Weight for 'In Forbidden Region' is strongly positive
        print(
            "\nConclusion: The model has successfully inferred a high cost for entering the 'Forbidden Region'."
        )
        print(
            "This indicates a learned behavioral constraint: 'Avoid the central 2x2 square'."
        )
    else:
        print(
            "\nConclusion: The model did not strongly infer the constraint. Try different initializations or more demonstrations."
        )


if __name__ == "__main__":
    run_inference()
