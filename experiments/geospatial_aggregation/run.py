import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import sys

# The test environment is expected to have the GA-sklearn repo
# source code available in the PYTHONPATH.
from model.estimator import GARegressor

def generate_synthetic_scene_data(num_scenes=50, max_objects=25, crowd_radius=0.15):
    """
    Generates a synthetic dataset mimicking object detection in scenes.
    Each object is a data point. The target variable is 'crowdedness',
    defined as the number of other objects within a specific radius.

    This tests if GeoAggregator can learn a spatial property.
    """
    all_objects = []
    for scene_id in range(num_scenes):
        num_objects = np.random.randint(2, max_objects + 1)
        # Generate object data with coordinates and a categorical feature
        objects = pd.DataFrame({
            'x': np.random.rand(num_objects),
            'y': np.random.rand(num_objects),
            'obj_type': np.random.randint(0, 5, size=num_objects),
            'scene_id': scene_id
        })

        # Calculate crowdedness for each object based on neighbors
        crowdedness = []
        for i, obj1 in objects.iterrows():
            count = 0
            for j, obj2 in objects.iterrows():
                if i == j:
                    continue
                dist = np.sqrt((obj1['x'] - obj2['x'])**2 + (obj1['y'] - obj2['y'])**2)
                if dist < crowd_radius:
                    count += 1
            crowdedness.append(count)
        objects['crowdedness'] = crowdedness
        all_objects.append(objects)

    return pd.concat(all_objects, ignore_index=True)


def main():
    """ Main experiment function """
    print("--- Running GeoAggregator Scene Crowdedness Experiment ---")

    # 1. Generate Data
    print("\n[Step 1/3] Generating synthetic scene data...")
    df = generate_synthetic_scene_data(num_scenes=200, max_objects=20, crowd_radius=0.2)
    print(f"Generated a dataset with {len(df)} total objects.")

    # Define columns for the model, similar to the source demo
    tab_x = ['obj_type']
    tab_l = ['x', 'y']
    tab_y = ['crowdedness']

    # 2. Train-Test Split
    X = df[tab_x + tab_l]
    y = df[tab_y]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 3. Configure and Train GeoAggregator
    print("\n[Step 2/3] Configuring and training GARegressor...")
    params = {
        'x_cols': tab_x,
        'spa_cols': tab_l,
        'y_cols': tab_y,
        'attn_variant': 'MCPA',
        'd_model': 32,
        'n_attn_layer': 2, # Use 2 layers to better capture spatial relations
        'idu_points': 1,
        'seq_len': 128, # A common sequence length from the demo
        'attn_dropout': 0.1,
        'attn_bias_factor': 5.0, # Add a Gaussian bias, a key feature of GA
        'reg_lin_dims': [16, 1],
        'epochs': 25,
        'lr': 5e-3,
        'batch_size': 32,
        'verbose': True
    }

    model = GARegressor(**params)
    model.fit(X=X_train[tab_x], l=X_train[tab_l], y=y_train)

    # 4. Evaluate the model
    print("\n[Step 3/3] Evaluating the model on the test set...")
    y_pred = model.predict(X=X_test[tab_x], l=X_test[tab_l])

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("\n--- Experiment Results ---")
    print(f"Target: Predicting 'crowdedness' of an object based on its neighbors.")
    print(f"R-squared: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print("--- Experiment Finished ---\n")

    if r2 > 0.1:
        print("Success: Model achieved R2 > 0.1, indicating it learned the spatial relationship.")
    else:
        print("Failure: Model did not learn the spatial relationship effectively (R2 <= 0.1).")


if __name__ == "__main__":
    main()
