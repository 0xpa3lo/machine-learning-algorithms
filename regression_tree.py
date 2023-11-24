class TreeNode:
    def __init__(self, examples):
        self.examples = examples
        self.left = None
        self.right = None
        self.split_point = None

    def split(self):
        # if len of examples is just 1 then we don't have to split
        if len(self.examples) < 2:
            return
        # everything we need to track for our split point
        best_split_point = {
            "feature": None,
            "value": None,  
            "mse": float("inf"),
            "split_index": None
        }
        # evaluate every feature and every split point in that feature
        for feature in self.examples[0].keys():
            # don't evaluate the label as a feature to split on
            if feature == "bpd":
                continue

            # sort example by our feature
            self.examples.sort(key=lambda examples: examples[feature])

            # evaluate all the sorted feature values as potential split point
             # iterates through all the examples expect the last one
            for i, _ in enumerate(self.examples[:-1]):
                # average of 2 adjacent feature values in sorted examples
                split_point_value = (self.examples[i][feature] + self.examples[i + 1][feature]) / 2
                mse, split_index = self.get_split_point_mse(feature, split_point_value)
            # after finding mse. ether our best split point so far is better then the curent we are evaluating or not
            # if it is better we need to update it to the lower
                if best_split_point["mse"] > mse:
                    best_split_point = {
                        "feature": feature,
                        "value": split_point_value,
                        "mse": mse,
                        "split_index": split_index
                    }
          # when we iterate throu all the feature and all the potential values of that feature
           # lets update the tree node with correct children the examples leading to the children and its own best split point
        self.split_point = best_split_point
          # that will sort the examples in this node by the best split point features
        self.examples.sort(key=lambda examples: examples[self.split_point["feature"]])
          # we do this that we can easily separeate left and right child examples
          # we assigne self.left to be a tree node with the examples up to split index
        self.left = TreeNode(self.examples[: self.split_point["split_index"]])
            # recursivly thoes what we did to this tree node with its left child
        self.left.split()
          # we assigne self.right to be a tree node with the examples to the right of the split index and inslude the split index
        self.right = TreeNode(self.examples[self.split_point["split_index"] :])
          # recusivly split the right child node
        self.right.split()

    def get_split_point_mse(self, feature, split_point_value):
   # we nee labled of examples in left and right child if we ware to split on these points
            # gets the lable for particular example only if the label is < of that split point values
        left_split_labels = [example["bpd"] for example in self.examples if example[feature] <= split_point_value]
        right_split_labels = [example["bpd"] for example in self.examples if example[feature] > split_point_value]
        
        # make sure that there are left and right examples. If not no split to evaluate
        if not len(left_split_labels) or not len(right_split_labels):
            return None, None
        left_split_mse = get_mse(left_split_labels)
        right_split_mse = get_mse(right_split_labels)
        
       # Total split point mse. In we could only added the mse of left and right nodes together. but doint for every node is good for diagnostic purpose
        num_samples = len(left_split_labels) + len(right_split_labels)
        mse = ((len(left_split_labels) * left_split_mse) + (len(right_split_labels) * right_split_mse)) / num_samples
       # if we sort our examples by particular feature anything on the left will be in the left child and on the right in the right child. just to know what is where
        split_index = len(left_split_labels)
        return mse, split_index

# sum of the squared diff between the value and the average of all the values
def get_mse(values):
    average = get_average(values)
    return sum([(value - average) ** 2 for value in values]) / len(values)

def get_average(values):
    return sum(values) / len(values)

class RegressionTree:
    def __init__(self, examples):
        self.root = TreeNode(examples)
        self.train()

    def train(self):
        self.root.split()

    def predict(self, example):
      # all: based on current node. if current examples feature is < value indicate by the split point of that node
        # move to the left or right. contine  until the node is a leafe node
        # crete a node 
        node = self.root
        # while this node has children traver left or right depending on current node split point
        while node.left and node.right:
           # if example that we provided in the input that example feature less or equla to split point 
            # assignet to that particular node. then assigne the node to be that node
            if example[node.split_point["feature"]] <= node.split_point['value']:
                node = node.left
            else:
                node = node.right
            # prediction. store all the labels of the examples in the leaf nodes
            # every element in the list is a label of particualr leaf eaxmple
        leaf_labels = [leaf_example["bpd"] for leaf_example in node.examples]
        # return the average of the leaf labels
        return sum(leaf_labels) / len(leaf_labels)
