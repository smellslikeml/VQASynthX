import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import torch


def main():
    """
    Main function to run the GeoAggregator experiment on synthetic 3D scene data.
    """
    print("--- Starting GeoAggregator for 3D Scene Regression Experiment ---")

    # This experiment assumes the GA-sklearn repository has been cloned and
    # is available in the PYTHONPATH.
    try:
        from model.estimator import GARegressor
    except ImportError:
        print("Error: Could not import GARegressor.")
        print(
            "Please ensure the 'GA-sklearn' repository is cloned and its path is added to PYTHONPATH."
        )
        print("Example: git clone https://github.com/ruid7181/GA-sklearn.git")
        print("         export PYTHONPATH=$(pwd)/GA-sklearn:$PYTHONPATH")
        sys.exit(1)

    # --- Step 1: Generate Synthetic 3D Scene Data ---
    print("\n[Step 1/4] Generating synthetic 3D scene data...")
    df = generate_synthetic_scene_data(num_objects=500, noise=0.1)

    tab_x = ["class_id", "obj_size"]
    tab_l = ["x", "y"]  # 2D spatial coordinates for the model
    tab_y = ["z"]  # Target variable: the z-coordinate (height)

    print(f"Generated a scene with {len(df)} objects.")
    print("Sample data:")
    print(df.head())

    # --- Step 2: Prepare Data for the Model ---
    print("\n[Step 2/4] Preparing data and splitting into train/test sets...")
    # The model expects normalized spatial coordinates
    df[tab_l] = df[tab_l].apply(lambda c: (c - c.min()) / (c.max() - c.min() + 1e-8))

    X_df, y_df = df[tab_x + tab_l], df[tab_y]
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, test_size=0.25, random_state=42
    )
    print(f"Train set: {len(X_train)} objects, Test set: {len(X_test)} objects.")

    # --- Step 3: Train the GeoAggregator Model ---
    print("\n[Step 3/4] Training the GeoAggregator model...")
    # Hyperparameters adapted from the GA-sklearn demo for this new task
    params = {
        "x_cols": tab_x,
        "spa_cols": tab_l,
        "y_cols": tab_y,
        "attn_variant": "MCPA",  # Cartesian Product Attention
        "d_model": 32,
        "n_attn_layer": 2,
        "idu_points": 4,
        "seq_len": 144,  # Should be >= sqrt(n_samples)
        "attn_dropout": 0.05,
        "reg_lin_dims": [16, 1],
        "epochs": 40,
        "lr": 1e-3,
        "batch_size": 16,
        "verbose": True,
    }

    model = GARegressor(**params)
    # GARegressor expects separate dataframes for features (X), spatial coords (l), and target (y)
    model.fit(X=X_train[tab_x], l=X_train[tab_l], y=y_train)
    print("Model training complete.")

    # --- Step 4: Evaluate the Model ---
    print("\n[Step 4/4] Evaluating the model on the test set...")
    y_pred = model.predict(X=X_test[tab_x], l=X_test[tab_l], verbose=False)

    r2 = r2_score(y_true=y_test[tab_y], y_pred=y_pred)
    mae = mean_absolute_error(y_true=y_test[tab_y], y_pred=y_pred)

    print("\n--- Experiment Results ---")
    print(
        f"Task: Predict an object's z-coordinate based on its (x, y) location and features."
    )
    print(f"R-squared: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print("--- Experiment Finished ---\n")


def generate_synthetic_scene_data(num_objects: int, noise: float = 0.1) -> pd.DataFrame:
    """
    Generates a DataFrame representing objects in a single synthetic 3D scene.

    In this scene, an object's height (z) is primarily a function of its
    (x, y) position, simulating objects resting on a sloped surface.

    Args:
        num_objects: The number of objects to generate in the scene.
        noise: The amount of random noise to add to the z-coordinate.

    Returns:
        A pandas DataFrame with object data.
    """
    # Define a sloped plane: z = 0.8*x + 0.3*y + 2
    x = np.random.rand(num_objects) * 10
    y = np.random.rand(num_objects) * 10
    z = 0.8 * x + 0.3 * y + 2 + np.random.randn(num_objects) * noise

    # Add other categorical and numerical features
    class_id = np.random.randint(0, 5, size=num_objects)  # e.g., 0:chair, 1:table
    obj_size = np.random.rand(num_objects) * 0.5 + 0.1  # represents object volume/size

    df = pd.DataFrame(
        {"x": x, "y": y, "z": z, "class_id": class_id, "obj_size": obj_size}
    )
    return df


if __name__ == "__main__":
    main()
