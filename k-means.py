import random  # Importing the random module for random number generation

class ClusterCenter:
    def __init__(self, coordinates):
        self.coordinates = coordinates  # Initializes the cluster center with a given coordinates (features)
        self.assigned_users = set()  # A set to keep track of users assigned to this cluster center

def perform_k_means(user_features, features_count, clusters_count):
    random.seed(42)  # Setting a fixed seed for random number generation to ensure reproducibility
    # Randomly selecting initial users to act as the initial cluster centers
    initial_cluster_users = random.sample(sorted(list(user_features.keys())), clusters_count)

    # Creating ClusterCenter objects for each selected initial user
    cluster_centers = [ClusterCenter(user_features[user]) for user in initial_cluster_users]

    for _ in range(10):  # Running the algorithm for a fixed number of iterations (10)
        for user_id, features in user_features.items():  # Iterating over each user
            min_distance = float("inf")  # Initializing the minimum distance as infinity
            nearest_center = None  # Placeholder for the nearest cluster center

            # Finding the nearest cluster center to the current user
            for center in cluster_centers:
                distance = calculate_manhattan_distance(features, center.coordinates)  # Calculating Manhattan distance
                if distance < min_distance:  # Checking if this center is nearer
                    min_distance = distance
                    nearest_center = center

            nearest_center.assigned_users.add(user_id)  # Assigning the user to the nearest cluster center

        # Updating the coordinates of each cluster center
        for center in cluster_centers:
            center.coordinates = compute_center_average(center, features_count, user_features)  # Recomputing center's coordinates
            center.assigned_users.clear()  # Clearing the set for the next iteration

    return [center.coordinates for center in cluster_centers]  # Returning the final coordinates of cluster centers

def calculate_manhattan_distance(features, other_features):
    # Calculating Manhattan distance between two sets of features
    differences = [abs(features[i] - other_features[i]) for i in range(len(features))]
    return sum(differences)  # Summing up the absolute differences

def compute_center_average(center, features_count, user_features):
    average_coordinates = [0] * features_count  # Initializing a list to store the average for each feature
    # Calculating the average for each feature
    for i in range(features_count):
        for user in center.assigned_users:
            average_coordinates[i] += user_features[user][i]  # Summing up the features of all assigned users
    # Dividing each feature sum by the number of assigned users to get the average
    return [value / len(center.assigned_users) if center.assigned_users else 0 for value in average_coordinates]
