# Machine learning algorithms


### To do:
1. Multinomial Naive Bayes

  Pseudo code:
    
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
      




KNN

   Prediction
    
      given features 
      for each pid:
        compute distance(features, pid['features'])
      sort(distances
      return most_frequent_label(dosatmces(I-> K))



K-means

    init centroid
    for num_iterations:
      for user in data:
        for centroid in centroids:
          min_distance --> centroid.add(user)
    
    for centroid in centroids:
      update_centroid_location
    return centroid_location 
  
Regression Tree

Basic neuron
