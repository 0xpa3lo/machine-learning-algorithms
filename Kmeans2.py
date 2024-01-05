import random

# iter number
# for user and centroid calculate Manhattan distance
# add user to appropriate centroid
# update centroid

class Centroid:
    def __init__(self, location):
        self.location = location
        self.closest_users = set()
      
def get_k_means(user_feature_map, num_features_per_user, k):
    random.seed(1337)
    # Gets the inital users, to be used as centroids
    initial_centroid_users = random.sample(sorted(list(user_feature_map.keys())), k)

    centroids = [Centroid(user_feature_map[inital_centroid_user]) for inital_centroid_user in initial_centroid_users]
    # 100 iterations 
    for _ in range(100):
      # loop thorugh users amd base on the features and manhattan distance find the closest centroid and add the user to the centroid
        for user_id, features in user_feature_map.items():
            closest_centroid = min(centroids, key=lambda centroid: get_manhattan_distance(centroid.location, features, num_features_per_user))
            closest_centroid.closest_users.add(user_id)
        # Update centroid location after we found the closes users
        for centroid in centroids:
            centroid.location = update_location(centroid, user_feature_map, num_features_per_user)
          # don't forget to clear tghe closest users list after updating location
            centroid.closest_users.clear()
    
    return [centroid.location for centroid in centroids]

def update_location(centroid, user_feature_map, num_features):
  # create a var that has number as many as we have features
    avg_location = [0] * num_features
  # loop through the users and for the feature at location i at to location i in avg_location
    for user_id in centroid.closest_users:
        for i in range(num_features):
            avg_location[i] += user_feature_map[user_id][i]
    # num of closes users per centroid  
    num_users = len(centroid.closest_users)
  # return average
    return [total / num_users for total in avg_location]
            
        
def get_manhattan_distance(point_a, point_b, num_features):
  # sum of |a-b| 
    return sum(abs(a - b) for a, b in zip(point_a, point_b))















