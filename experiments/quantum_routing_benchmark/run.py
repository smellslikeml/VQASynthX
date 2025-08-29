# experiments/quantum_routing_benchmark/run.py
#
# This script reproduces benchmark results from the Quantum-AI-Network-Optimization
# project (https://github.com/ailabteam/Quantum-AI-Network-Optimization).
#
# Its purpose is to document the observed performance limitations of Variational
# Quantum Eigensolver (VQE) and Quantum Approximate Optimization Algorithm (QAOA)
# for a shortest-path network problem.
#
# The hardcoded data shows that VQE converges to an invalid solution and QAOA
# exhibits unstable, non-convergent behavior. These findings justify prioritizing
# classical optimization algorithms for tasks within the VQASynth project.

import matplotlib.pyplot as plt
import numpy as np
import os

# --- Global plot settings for consistency ---
plt.style.use("seaborn-v0_8-whitegrid")
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 12


def generate_quantum_optimizer_plots():
    """
    Generates and saves convergence plots for VQE and QAOA based on
    pre-computed experimental data, including text annotations that
    summarize the outcome.
    """
    print("Generating plots for VQE and QAOA convergence benchmarks...")

    # --- Data from Quantum-AI-Network-Optimization experiments ---
    vqe_steps = [
        20,
        40,
        60,
        80,
        100,
        120,
        140,
        160,
        180,
        200,
        220,
        240,
        260,
        280,
        300,
        320,
        340,
        360,
        380,
        400,
    ]
    vqe_energy = [
        216.75,
        184.09,
        170.34,
        164.46,
        162.83,
        162.53,
        162.503,
        162.5003,
        162.50005,
        162.50000,
        162.50000,
        162.50000,
        162.50000,
        162.50000,
        162.50000,
        162.50000,
        162.50000,
        162.504,
        162.5000,
        162.5000,
    ]

    qaoa_steps = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]
    qaoa_energy = [
        322.10,
        330.77,
        324.50,
        390.31,
        341.56,
        416.75,
        337.08,
        309.91,
        326.20,
        317.49,
        322.82,
        275.55,
        283.64,
        259.38,
        355.62,
    ]

    # --- Setup output directory ---
    output_dir = os.path.join("results", "figures")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Ensured output directory exists: '{output_dir}'")

    # --- Plot 1: VQE Convergence ---
    plt.figure(figsize=(8, 6))
    ax1 = plt.gca()
    ax1.plot(vqe_steps, vqe_energy, marker="o", linestyle="-", color="royalblue")
    ax1.set_title(
        "VQE Convergence Benchmark for Shortest Path", fontsize=TITLE_FONTSIZE
    )
    ax1.set_xlabel("Optimization Steps", fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel("Energy (Cost Function Value)", fontsize=LABEL_FONTSIZE)
    ax1.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Text annotation summarizing the result
    ax1.text(
        0.5,
        0.5,
        "Converged to an\nInvalid Solution State",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax1.transAxes,
        fontsize=14,
        color="darkred",
        weight="bold",
    )

    # Save the VQE figure
    vqe_output_path = os.path.join(output_dir, "vqe_sp_convergence.png")
    plt.savefig(vqe_output_path, dpi=300, bbox_inches="tight")
    print(f"VQE plot saved to '{vqe_output_path}'")
    plt.close()

    # --- Plot 2: QAOA Convergence ---
    plt.figure(figsize=(8, 6))
    ax2 = plt.gca()
    ax2.plot(qaoa_steps, qaoa_energy, marker="x", linestyle="--", color="crimson")
    ax2.set_title(
        "QAOA Convergence Benchmark for Shortest Path", fontsize=TITLE_FONTSIZE
    )
    ax2.set_xlabel("Optimization Steps", fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel("Energy (Cost Function Value)", fontsize=LABEL_FONTSIZE)
    ax2.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Text annotation summarizing the result
    ax2.text(
        0.5,
        0.5,
        "Non-Convergent Behavior\n(Indicates Barren Plateau)",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax2.transAxes,
        fontsize=14,
        color="darkred",
        weight="bold",
    )

    # Save the QAOA figure
    qaoa_output_path = os.path.join(output_dir, "qaoa_sp_convergence.png")
    plt.savefig(qaoa_output_path, dpi=300, bbox_inches="tight")
    print(f"QAOA plot saved to '{qaoa_output_path}'")
    plt.close()


def main():
    """Main function to run the plot generation."""
    generate_quantum_optimizer_plots()
    print("\nBenchmark plot generation complete.")


if __name__ == "__main__":
    main()
