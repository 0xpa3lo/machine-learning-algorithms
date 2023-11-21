import random
import math

class Centroid:
    def __init__(self, location):
        self.location = location
        self.closest_users = set()


def get_k_means(user_feature_map, num_features_per_user, k):
    # Don't change the following two lines of code.
    random.seed(42)
    # Gets the inital users, to be used as centroids.
    inital_centroid_users = random.sample(sorted(list(user_feature_map.keys())), k)

    centroids = [Centroid(user_feature_map[inital_centroid_user])for inital_centroid_user in inital_centroid_users]

    for i in range(10):
        for uid, features in user_feature_map.items():
            closest_centroid_distance = float("inf")
            closest_centroid = None
            for centroid in centroids:
                distance = calc_manhattan_distance(centroid, uid, num_features_per_user)
                closest_centroid_distance = distance
                if closest_centroid_distance < distance:
                    closest_centroid_distance = distance
                    closest_centroid = centroid
                else:
                    closest_centroid = centroid
            self.closest_users.add(uid)
            
        for centroid in centroid:
            centroid.location = get_centroid_avg(centroid, self.closes_users)
            self.closest_users.clear()
    return [centroid.location for centroid in centroids]
                
def calc_manhattan_distance(centroid, user, num_features_per_user):
    distance = 0
    for x in num_features:
        distance += abs(centroid[x] - user[x])
    return distance

def get_centroid_avg(centroid, closest_users, num_features_per_user):
    centroid_avg_distance = [0 for i in num_features_per_user]
    for user in closest_users:
        for i in num_features_per_user:
            centroid_avg_distance[i] += user[i]

    for i in num_features_per_user:
        centroid_avg_distance[i] = centroid_avg_distance[i] / num_features_per_user
        
    return centroid_avg_distance
    
  
