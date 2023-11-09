# Machine learning algorithms

### Purpose
Basic machine learning algorithms practice

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

K-means

Regression Tree

Basic neuron
