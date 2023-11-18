import math

def predict_label(examples, features, k, label_key="is_intrusive"):
    """
    Predicts the label for a given set of features using the K-Nearest Neighbors algorithm.
    This function finds the k nearest neighbors from the given examples using the Euclidean distance.
    It then calculates the majority label from these neighbors as the predicted label for the input features.

    Args:
        examples (dict): A dictionary of examples with their feature values and labels.
        features (list): The feature list of the new example whose label needs to be predicted.
        k (int): The number of nearest neighbors to consider.
        label_key (str, optional): The key in the examples dictionary for the label. Defaults to "is_intrusive".

    Returns:
        int: The predicted label, 0 or 1, based on the majority voting of k-nearest neighbors.
    """
    knn = find_k_nearest_neighbors(examples, features, k)
    knn_labels = [examples[pid][label_key] for pid in knn]
    return round(sum(knn_labels)/k)

def find_k_nearest_neighbors(examples, features, k):
    """
    Finds the k nearest neighbors of a given data point in the feature space.
    This function calculates the Euclidean distance between the input features and each example in the dataset.
    It returns the identifiers of the k examples with the smallest distances.

    Args:
        examples (dict): A dictionary of examples with their feature values and labels.
        features (list): The feature list of the new example.
        k (int): The number of nearest neighbors to find.

    Returns:
        list: A list of identifiers (keys) for the k nearest neighbors.
    """
    distances = {}
    for pid, features_label_map in examples.items():
        distance = get_euclidean_distance(features, features_label_map['features'])
        distances[pid] = distance

    return sorted(distances, key=distances.get)[:k]

def get_euclidean_distance(features, other_features):
    """
    Calculates the Euclidean distance between two data points in the feature space.

    Args:
        features (list): The feature list of the first data point.
        other_features (list): The feature list of the second data point.

    Returns:
        float: The Euclidean distance between the two data points.
    """
    squared_differences = []
    for i in range(len(features)):
        squared_differences.append((other_features[i] - features[i])**2)
    
    return math.sqrt(sum(squared_differences))
