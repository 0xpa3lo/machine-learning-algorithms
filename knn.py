import math

def predict_label(data_points, target_features, neighbors_count, label_key="is_intrusive"):
    """
    Uses K-Nearest Neighbors to predict the label of a new data point.
    Finds the 'neighbors_count' nearest data points and uses majority vote for prediction.

    Args:
        data_points (dict): Collection of known data points with features and labels.
        target_features (list): Features of the data point to be labeled.
        neighbors_count (int): Number of nearest neighbors to consider for prediction.
        label_key (str, optional): Key for the label in data_points. Defaults to "is_intrusive".

    Returns:
        int: Predicted label (0 or 1) based on nearest neighbors.
    """
    nearest_neighbors = find_k_nearest_neighbors(data_points, target_features, neighbors_count)
    neighbor_labels = [data_points[point_id][label_key] for point_id in nearest_neighbors]
    return round(sum(neighbor_labels) / neighbors_count)

def find_k_nearest_neighbors(data_points, target_features, neighbors_count):
    """
    Identifies the nearest neighbors of a new data point in the dataset.
    Calculates Euclidean distance to each data point and picks the closest ones.

    Args:
        data_points (dict): Known data points with features and labels.
        target_features (list): Features of the new data point.
        neighbors_count (int): Number of nearest neighbors to find.

    Returns:
        list: Identifiers for the nearest neighbors.
    """
    distances = {}
    for point_id, feature_label_map in data_points.items():
        distance = calculate_euclidean_distance(target_features, feature_label_map['features'])
        distances[point_id] = distance

    return sorted(distances, key=distances.get)[:neighbors_count]

def calculate_euclidean_distance(target_features, comparison_features):
    """
    Computes the Euclidean distance between two feature sets.

    Args:
        target_features (list): Feature set of the first data point.
        comparison_features (list): Feature set of the second data point.

    Returns:
        float: Euclidean distance between the two feature sets.
    """
    squared_differences = [(comparison_features[i] - target_features[i])**2 for i in range(len(target_features))]
    
    return math.sqrt(sum(squared_differences))
