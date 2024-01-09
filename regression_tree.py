from collections import defaultdict
import math

class DecisionNode:
    def __init__(self, data, target):
        self.data = data
        self.left_child = None
        self.right_child = None
        self.best_split = None
        self.target = target

    def split_node(self):
        # Stop splitting if there's only one or no data point
        if len(self.data) < 2:
            return

        optimal_split = {
            "feature": None,
            "threshold": None,  
            "mse": float("inf"),
            "index": None
        }

        # Evaluate each feature for a potential split
        for feature in self.data[0].keys():
            if feature == "bpd":  # Skip the label
                continue

            self.data.sort(key=lambda item: item[feature])

            for i in range(len(self.data) - 1):
                split_val = (self.data[i][feature] + self.data[i + 1][feature]) / 2
                mse, index = self.calculate_split_mse(feature, split_val)
                if mse < optimal_split["mse"]:
                    optimal_split = {
                        "feature": feature,
                        "threshold": split_val,
                        "mse": mse,
                        "index": index
                    }

        self.best_split = optimal_split
        self.data.sort(key=lambda item: item[self.best_split["feature"]])

        self.left_child = DecisionNode(self.data[: self.best_split["index"]])
        self.left_child.split_node()

        self.right_child = DecisionNode(self.data[self.best_split["index"] :])
        self.right_child.split_node()

    def calculate_split_mse(self, feature, split_val):
        left_labels = [item[self.target] for item in self.data if item[feature] <= split_val]
        right_labels = [item[self.target] for item in self.data if item[feature] > split_val]

        if not left_labels or not right_labels:
            return None, None

        left_mse = calculate_mse(left_labels)
        right_mse = calculate_mse(right_labels)

        total_samples = len(left_labels) + len(right_labels)
        combined_mse = ((len(left_labels) * left_mse) + (len(right_labels) * right_mse)) / total_samples
        index = len(left_labels)
        return combined_mse, index

def calculate_mse(values):
    mean = calculate_mean(values)
    return sum([(value - mean) ** 2 for value in values]) / len(values)

def calculate_mean(values):
    return sum(values) / len(values)

class RegressionTree:
    def __init__(self, data, target):
        self.root = DecisionNode(data)
        self.build_tree()
        self.target = target 
        
    def build_tree(self):
        self.root.split_node()

    def predict(self, sample):
        current_node = self.root
        while current_node.left_child and current_node.right_child:
            if sample[current_node.best_split["feature"]] <= current_node.best_split['threshold']:
                current_node = current_node.left_child
            else:
                current_node = current_node.right_child

        leaf_labels = [leaf[target] for leaf in current_node.data]
        return calculate_mean(leaf_labels)
