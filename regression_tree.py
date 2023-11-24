class TreeNode:
    def __init__(self, examples):
        self.examples = examples
        self.left = None
        self.right = None
        self.split_point = None

    def split(self):
        # Write your code here.
        pass


class RegressionTree:
    def __init__(self, examples):
        # Don't change the following two lines of code.
        self.root = TreeNode(examples)
        self.train()

    def train(self):
        # Don't edit this line.
        self.root.split()

    def predict(self, example):
        # Write your code here.
        return 0
