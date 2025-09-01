import argparse
import os
import pandas as pd
import numpy as np
import plotly.express as px
import scipy.stats as stats
import re
from pathlib import Path

# --- Functions adapted from cadavid1/PCF/Experiment_Analysis.py ---


def perform_t_test(data1, data2):
    """Perform a two-sample t-test and return the t-statistic and p-value."""
    # This function is directly inspired by the PCF source code.
    t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False, nan_policy="omit")
    return t_stat, p_value


def clean_column_names(df):
    """Standardize column names. Inspired by PCF's data hygiene approach."""
    # This function is directly inspired by the PCF source code.
    df.columns = df.columns.str.strip().str.lower()
    df.columns = df.columns.str.replace(r"\W+", "_", regex=True)
    return df


# --- Main analysis logic for VQASynth pipeline ---


def analyze_datasets(input_dir, output_dir, column_to_analyze):
    """
    Loads datasets, categorizes them by filename, performs analysis, and saves results.
    """
    print(f"Starting analysis on directory: {input_dir}")
    dataframes = []

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Use filename-based categorization, inspired by PCF
    files = [f for f in os.listdir(input_dir) if f.endswith(".jsonl")]
    if not files:
        print("No .jsonl files found in the input directory. Exiting.")
        return

    for filename in files:
        try:
            # Assume category is the part of the filename before the first underscore or dot
            category_match = re.match(r"^([a-zA-Z0-9_-]+?)_?.*\.jsonl", filename)
            category = category_match.group(1) if category_match else "unknown"

            file_path = os.path.join(input_dir, filename)
            df = pd.read_json(file_path, lines=True)
            df["category"] = category
            dataframes.append(df)
            print(f"Loaded {filename} with category '{category}'")
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

    if not dataframes:
        print("No dataframes were loaded. Exiting.")
        return

    # Combine into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df = clean_column_names(combined_df)

    # Ensure the analysis column exists
    if column_to_analyze not in combined_df.columns:
        print(f"Error: Column '{column_to_analyze}' not found in the combined data.")
        print(f"Available columns: {list(combined_df.columns)}")
        return

    categories = combined_df["category"].unique()
    results_summary = []

    # Perform pairwise t-tests, similar to PCF's `calculate_significance_across_categories`
    if len(categories) >= 2:
        print(f"\nPerforming pairwise t-tests on column: '{column_to_analyze}'")
        for i, cat1 in enumerate(categories):
            for cat2 in categories[i + 1 :]:
                data_cat1 = combined_df[combined_df["category"] == cat1][
                    column_to_analyze
                ].dropna()
                data_cat2 = combined_df[combined_df["category"] == cat2][
                    column_to_analyze
                ].dropna()

                if not data_cat1.empty and not data_cat2.empty:
                    t_stat, p_value = perform_t_test(data_cat1, data_cat2)
                    result_line = f"T-test between '{cat1}' and '{cat2}': t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}"
                    print(result_line)
                    results_summary.append(result_line)

        # Generate and save a box plot for visualization
        try:
            fig = px.box(
                combined_df,
                x="category",
                y=column_to_analyze,
                title=f"Distribution of '{column_to_analyze}' by Category",
                points="all",
            )
            plot_path = os.path.join(output_dir, f"{column_to_analyze}_comparison.png")
            fig.write_image(plot_path)
            print(f"\nSaved box plot to {plot_path}")
            results_summary.append(f"\nComparison plot saved to: {plot_path}")
        except Exception as e:
            print(f"Could not generate plot: {e}")

    else:
        result_line = "Less than two categories found, skipping pairwise comparison."
        print(result_line)
        results_summary.append(result_line)

    # Save summary report
    report_path = os.path.join(output_dir, "analysis_report.txt")
    with open(report_path, "w") as f:
        f.write("VQASynth Dataset Analysis Report\n")
        f.write("=================================\n\n")
        f.write("\n".join(results_summary))
    print(f"Saved analysis report to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and compare VQASynth datasets."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing .jsonl dataset files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save analysis results.",
    )
    parser.add_argument(
        "--column",
        type=str,
        default="distance_meters",
        help="The numerical column to analyze and compare across datasets.",
    )
    args = parser.parse_args()

    analyze_datasets(args.input_dir, args.output_dir, args.column)


if __name__ == "__main__":
    main()
