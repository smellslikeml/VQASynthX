#!/usr/bin/env python
import argparse
import json
import math
import torch
import numpy as np
from tqdm import tqdm

# --- Helper functions adapted from SOURCE repo (x1ng5/gd_sampling) ---
# These functions are based on the provided gd_iter.ipynb and the Docker demo script.


def generate_x(example_num, x_dim, batch_num, data_range, xi, device):
    """Generates context features `x`."""
    x_all = torch.randn(batch_num, example_num, x_dim, device=device) * data_range
    x_all = x_all / x_all.norm(dim=-1, keepdim=True)
    return x_all


def generate_y(x_all, weight_range, noise, device, few_shot, choose_elements):
    """Generates context labels `y` and the true weight vector `w`."""
    b, n, d = x_all.size()
    w_all = torch.randn(b, d, device=device) * weight_range
    w_all = w_all / w_all.norm(dim=-1, keepdim=True)

    if few_shot:
        mask = torch.zeros(b, d, device=device)
        # Ensure repeatable sparse masks for a given batch size if needed, but randperm is fine here
        indices = torch.cat(
            [torch.randperm(d)[:choose_elements] for _ in range(b)]
        ).reshape(b, choose_elements)
        mask.scatter_(1, indices, 1)
        w_all = w_all * mask

    y_all = (x_all @ w_all[:, :, None]).squeeze(-1) + torch.randn(
        b, n, device=device
    ) * noise
    return y_all, w_all


def conduct_gd_noise(x, y, eta, steps, lbda, gd_noise, gd_noise_func, w0=None):
    """Performs gradient descent with optional L2 regularization and/or injected noise."""
    b, n, d = x.size()
    if w0 is None:
        wi = torch.zeros(b, 1, d, device=x.device)
    else:
        wi = w0.clone()

    w_cot = torch.zeros(b, steps, d, device=x.device)

    for t in range(steps):
        # Standard linear regression gradient for MSE
        grad = -2 * ((y - (x @ wi.transpose(1, 2)).squeeze(-1))[:, :, None] * x).mean(
            dim=1, keepdim=True
        )
        grad += 2 * lbda * wi
        if gd_noise > 0:
            grad += gd_noise_func(gd_noise, wi, x.device)
        wi = wi - eta * grad
        w_cot[:, t, :] = wi.squeeze(1)

    return w_cot


def gd_noise_nwn(gd_noise, wi, device):
    """Specific noise injection function from the SOURCE repo."""
    gd_noise_v = torch.randn(wi.size(), device=device) * gd_noise
    return (gd_noise_v.transpose(1, 2) @ wi) * gd_noise_v


def format_question(context_pairs, query_x):
    """Formats the regression problem into a natural language question."""
    header = "Given the following (X, Y) pairs, where X is a vector and Y is a scalar:"
    examples = []
    for x, y in context_pairs:
        x_str = ", ".join([f"{val:.4f}" for val in x])
        y_str = f"{y.item():.4f}"
        examples.append(f"X = [{x_str}], Y = {y_str}")

    query_x_str = ", ".join([f"{val:.4f}" for val in query_x])
    question = f"What is the predicted Y value for X = [{query_x_str}]?"

    return f"{header}\n\n" + "\n".join(examples) + f"\n\n{question}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate data for in-context linear regression task."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of VQA samples to generate.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="in_context_regression_data.jsonl",
        help="Path to the output JSONL file.",
    )
    # Hyperparameters inspired by SOURCE repo
    parser.add_argument(
        "--x-dim", type=int, default=10, help="Dimension of the feature vector x."
    )
    parser.add_argument(
        "--example-num", type=int, default=20, help="Number of in-context examples."
    )
    parser.add_argument(
        "--choose-elements",
        type=int,
        default=3,
        help="Number of non-zero elements in the true weight vector (sparsity).",
    )
    parser.add_argument(
        "--noise", type=float, default=math.sqrt(0.1), help="Noise level in y labels."
    )
    parser.add_argument(
        "--gd-noise",
        type=float,
        default=1.0,
        help="Noise level injected into gradient descent.",
    )
    parser.add_argument(
        "--cot-steps", type=int, default=1000, help="Number of gradient descent steps."
    )
    parser.add_argument(
        "--sampling-runs",
        type=int,
        default=10,
        help="Number of noisy GD runs to average for the final answer.",
    )
    parser.add_argument("--eta", type=float, default=1e-3, help="Learning rate for GD.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(args.output_file, "w") as f:
        for i in tqdm(range(args.num_samples), desc="Generating samples"):
            # We generate a single batch element (batch_num=1) for each sample
            batch_num = 1

            # 1. Generate context data and a query point
            x_context = generate_x(
                example_num=args.example_num,
                x_dim=args.x_dim,
                batch_num=batch_num,
                data_range=1,
                xi=0,
                device=device,
            )
            y_context, _ = generate_y(
                x_all=x_context,
                weight_range=1,
                noise=args.noise,
                device=device,
                few_shot=True,
                choose_elements=args.choose_elements,
            )
            x_query = generate_x(
                example_num=1,
                x_dim=args.x_dim,
                batch_num=batch_num,
                data_range=1,
                xi=0,
                device=device,
            )

            # 2. Solve for 'w' using averaged noisy GD, as in the source notebook
            w_samples = torch.zeros(
                batch_num, args.cot_steps, args.x_dim, device=device
            )
            for _ in range(args.sampling_runs):
                w_cot = conduct_gd_noise(
                    x_context,
                    y_context,
                    args.eta,
                    w0=None,
                    steps=args.cot_steps,
                    lbda=0,
                    gd_noise=math.sqrt(args.gd_noise),
                    gd_noise_func=gd_noise_nwn,
                )
                w_samples += w_cot

            w_final = (w_samples / args.sampling_runs)[
                :, -1, :
            ]  # Get the final weight vector from the last step

            # 3. Predict the answer for the query point
            y_query_pred = (x_query @ w_final.unsqueeze(-1)).item()

            # 4. Format for VQA
            context_pairs = list(
                zip(
                    x_context.squeeze(0).cpu().numpy(),
                    y_context.squeeze(0).cpu().numpy(),
                )
            )
            query_x_np = x_query.squeeze(0).squeeze(0).cpu().numpy()

            question = format_question(context_pairs, query_x_np)
            answer = y_query_pred

            # 5. Write to file in a standard instruction-tuning format
            record = {
                "id": f"in_context_regression_{i}",
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": f"{answer:.6f}"},
                ],
            }
            f.write(json.dumps(record) + "\n")

    print(f"Successfully generated {args.num_samples} samples to {args.output_file}")


if __name__ == "__main__":
    main()
