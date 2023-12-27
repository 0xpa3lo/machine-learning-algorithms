import numpy as np
import math

class Neuron:
    def __init__(self, examples):
        np.random.seed(42)
        # Three weights: one for each feature and one more for the bias.
        self.weights = np.random.normal(0, 1, 3 + 1)
        self.examples = examples
        self.train()

    def train(self, learning_rate=0.01, batch_size=10, epochs=200):
        for _ in range(epochs):
            for batch_index in range(0, len(self.examples), batch_size):
                # sub group of examples and compute weights 
                mini_batch = self.examples[batch_index:batch_index + batch_size]
                prediction_labels = [
                    {"prediction": self.predict(example['features']), "actual": example['label']}
                    for example in mini_batch
                ]
                gradients = self.__get_gradients(mini_batch, prediction_labels)
                # Update weights using NumPy for element-wise operations
                self.weights -= learning_rate * np.array(gradients)

    def predict(self, features):
        # Adding bias term to the input features
        model_inputs = np.append(features, 1)
        # Use numpy dot product for wTx calculation
        wTx = np.dot(self.weights, model_inputs)
        # Apply the sigmoid function
        return 1 / (1 + math.exp(-wTx))

    def __get_gradients(self, mini_batch, prediction_labels):
        # Calculate errors for each example
        errors = np.array([prediction_label['prediction'] - prediction_label['actual'] for prediction_label in prediction_labels])
        # Initialize gradients to zero
        gradients = np.zeros_like(self.weights)
        
        # Accumulate gradients for each example
        for example, error in zip(mini_batch, errors):
            # Adding bias term to the input features
            features = np.append(example['features'], 1)
            # Calculate gradients for current example and accumulate
            gradients += features * error
        
        # Average gradients over the batch
        gradients /= len(mini_batch)
        return gradients

