import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D


def generate_synthetic_point_cloud(n_points_cluster=300, n_points_noise=100):
    """
    Generates a synthetic 3D point cloud with a dense cluster and random noise.

    Returns:
        np.ndarray: A (N, 3) numpy array of points.
    """
    # Create a dense spherical cluster
    radius = 0.5
    phi = np.random.uniform(0, 2 * np.pi, n_points_cluster)
    costheta = np.random.uniform(-1, 1, n_points_cluster)
    u = np.random.uniform(0, 1, n_points_cluster)

    theta = np.arccos(costheta)
    r = radius * np.cbrt(u)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    cluster_points = np.vstack([x, y, z]).T
    cluster_points += np.array([1, 1, 1])  # Offset the cluster center

    # Create random background noise
    noise_points = np.random.uniform(-2, 2, size=(n_points_noise, 3))

    return np.vstack([cluster_points, noise_points])


def apply_dbscan(points, eps=0.4, min_samples=5, xy_weight=1.0, z_weight=1.0):
    """
    Applies DBSCAN clustering to a set of 3D points.
    This function is inspired by the data processing pipeline for mmWave radar data.
    It includes an optional weighting for different spatial dimensions.

    Args:
        points (np.ndarray): The input point cloud data (N, 3).
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        xy_weight (float): A weight applied to the X and Y coordinates to adjust their importance in clustering.
        z_weight (float): A weight applied to the Z coordinate.

    Returns:
        np.ndarray: An array of cluster labels for each point. Noise points are labeled -1.
    """
    if points.shape[0] == 0:
        return np.array([])

    weighted_data = points.copy()
    weighted_data[:, :2] *= xy_weight
    weighted_data[:, 2] *= z_weight

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(weighted_data)
    return clustering.labels_


def plot_clustering_results(points, labels):
    """
    Visualizes the original point cloud and the results of DBSCAN clustering.
    """
    fig = plt.figure(figsize=(18, 6))

    # Plot 1: Original Data
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], s=10, c="gray")
    ax1.set_title("Original Synthetic Point Cloud")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Plot 2: Clustered Data
    ax2 = fig.add_subplot(122, projection="3d")

    unique_labels = set(labels)
    colors = plt.cm.get_cmap("Spectral")(np.linspace(0, 1, len(unique_labels)))

    # Identify the largest cluster (main object)
    if len(unique_labels) > 1:
        # Exclude noise label (-1) for finding the largest cluster
        cluster_labels = [l for l in unique_labels if l != -1]
        if cluster_labels:
            largest_cluster_label = max(
                cluster_labels, key=lambda l: np.sum(labels == l)
            )
        else:
            largest_cluster_label = -2  # No non-noise clusters found
    else:
        largest_cluster_label = -2  # Only noise or one cluster

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Noise points in black
            class_member_mask = labels == k
            xy = points[class_member_mask]
            ax2.scatter(xy[:, 0], xy[:, 1], xy[:, 2], s=10, c="black", label="Noise")
        elif k == largest_cluster_label:
            # Largest cluster points in a distinct color
            class_member_mask = labels == k
            xy = points[class_member_mask]
            ax2.scatter(
                xy[:, 0], xy[:, 1], xy[:, 2], s=20, c="red", label="Primary Cluster"
            )
        else:
            # Other, smaller clusters
            class_member_mask = labels == k
            xy = points[class_member_mask]
            ax2.scatter(
                xy[:, 0], xy[:, 1], xy[:, 2], s=15, c=[col], label=f"Cluster {k}"
            )

    ax2.set_title("Point Cloud After DBSCAN Filtering")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("dbscan_clustering_results.png")
    print("Saved clustering visualization to dbscan_clustering_results.png")
    plt.show()


if __name__ == "__main__":
    print("Generating synthetic point cloud data...")
    point_cloud = generate_synthetic_point_cloud(
        n_points_cluster=400, n_points_noise=150
    )

    print("Applying DBSCAN for noise reduction...")
    # Parameters are tuned for the synthetic data generation function
    # The `eps` value is critical for defining the neighborhood size.
    labels = apply_dbscan(point_cloud, eps=0.2, min_samples=10)

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    num_noise = np.sum(labels == -1)
    print(f"DBSCAN found {num_clusters} clusters and {num_noise} noise points.")

    print("Visualizing results...")
    plot_clustering_results(point_cloud, labels)
