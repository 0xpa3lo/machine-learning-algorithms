import random  # Importing the random module for random number generation

class Centroid:
    def __init__(self, location):
        self.location = location  # Initializes the centroid with a given location (features)
        self.closest_users = set()  # A set to keep track of users closest to this centroid

def get_k_means(user_feature_map, num_features_per_user, k):
    random.seed(42)  # Setting a fixed seed for random number generation to ensure reproducibility
    # Randomly selecting initial users to act as the first centroids
    initial_centroid_users = random.sample(sorted(list(user_feature_map.keys())), k)

    # Creating Centroid objects for each selected initial user
    centroids = [Centroid(user_feature_map[initial_centroid_user]) for initial_centroid_user in initial_centroid_users]

    for _ in range(10):  # Running the algorithm for a fixed number of iterations (10)
        for uid, features in user_feature_map.items():  # Iterating over each user
            closest_centroid_distance = float("inf")  # Initializing the closest distance as infinity
            closest_centroid = None  # Placeholder for the closest centroid

            # Finding the closest centroid to the current user
            for centroid in centroids:
                distance = get_manhattan_distance(features, centroid.location)  # Calculating Manhattan distance
                if distance < closest_centroid_distance:  # Checking if this centroid is closer
                    closest_centroid_distance = distance
                    closest_centroid = centroid

            closest_centroid.closest_users.add(uid)  # Assigning the user to the closest centroid

        # Updating the location of each centroid
        for centroid in centroids:
            centroid.location = get_centroid_average(centroid, num_features_per_user, user_feature_map)  # Recomputing centroid's location
            centroid.closest_users.clear()  # Clearing the set for the next iteration

    return [centroid.location for centroid in centroids]  # Returning the final locations of centroids

def get_manhattan_distance(features, other_features):
    # Calculating Manhattan distance between two sets of features
    absolute_differences = [abs(features[i] - other_features[i]) for i in range(len(features))]
    return sum(absolute_differences)  # Summing up the absolute differences

def get_centroid_average(centroid, num_features_per_user, user_feature_map):
    centroid_average = [0] * num_features_per_user  # Initializing a list to store the average for each feature
    # Calculating the average for each feature
    for i in range(num_features_per_user):
        for user in centroid.closest_users:
            centroid_average[i] += user_feature_map[user][i]  # Summing up the features of all closest users
    # Dividing each feature sum by the number of closest users to get the average
    return [value / len(centroid.closest_users) if centroid.closest_users else 0 for value in centroid_average]
