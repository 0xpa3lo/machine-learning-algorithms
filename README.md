# Machine learning algorithms


### To do:
1. Multinomial Naive Bayes

    
    Training
    
      for tag: 
      log_probability(tag) = count of articles in tag / all articles across all tags
      
      for tag:
        for article:
          for word:
            log_probability(words | tag) = count of word in tag / count of word across tag


    Prediction

      for word in words:
        for tag:
          log_probability(tag | words) = log_probability(words | tag) + log_probability(tag)
      
  Time complexity: O(m*d*c) 
  n - number of samples
  c - number of classes
  d - number of dimensions


KNN

   Prediction
    
      given features 
      for each pid:
        compute distance(features, pid['features'])
      sort(distances
      return most_frequent_label(dosatmces(I-> K))
  
  Time complexity: O(n*d) 
  n - number of samples
  d - number of dimensions

K-means

    init centroid
    for num_iterations:
      for user in data:
        for centroid in centroids:
          min_distance --> centroid.add(user)
    
    for centroid in centroids:
      update_centroid_location
    return centroid_location 

  Time complexity: O(n*k*d) 
  n - number of iterations
  k - number of centroids
  d - number of dimensions
  
Regression Tree

Basic neuron
