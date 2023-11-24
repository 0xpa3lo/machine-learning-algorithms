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

  Training

        for each feature
          for each split_point_value
            best_split = min_mse(feature, split_point_value)
        split_examples_left()
        recurse()
        split_examples_right()
        recurse()

  Prediction

        tree_node = root
        while tree_node has children:
          if example[root.split_point['feature']] <= root.split_point['value']:
            tree_node = root.left
          else:
            tree_node = root.right

O(N * (N log N + F * N)), where N is the number of examples and F is the number of features. This is because the recursion (depth up to N) multiplies with the complexity of operations performed at each recursive call.
          




Basic neuron
