#!/usr/bin/env python3
"""2. Build a decision tree"""

import numpy as np


class Node:
    """Class Node that represents a node in a decision tree"""
    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Method that returns the maximum depth of the tree below the node"""
        if self.is_leaf:
            return self.depth
        else:
            return max(self.left_child.max_depth_below(),
                       self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """Method that returns the number of nodes below the node"""
        count = 0 if only_leaves else 1
        if self.left_child:
            count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child:
            count += self.right_child.count_nodes_below(only_leaves)
        return count

    def __str__(self):
        """Visual representation of the node and its children"""
        if self.is_root:
            out = (f"root [feature={self.feature}, "
                   f"threshold={self.threshold}]\n")
        else:
            out = (f"-> node [feature={self.feature}, "
                   f"threshold={self.threshold}]\n")

        if self.left_child:
            out += self.left_child_add_prefix(str(self.left_child))
        if self.right_child:
            out += self.right_child_add_prefix(str(self.right_child))
        return out

    def left_child_add_prefix(self, text):
        """Method that adds a prefix to the left child of the node"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Method that adds a prefix to the right child of the node"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("       " + x) + "\n"
        return new_text

    def get_leaves_below(self):
        """Method that returns a list of the leaves below the node"""
        leaves = []
        if self.left_child:
            leaves += self.left_child.get_leaves_below()
        if self.right_child:
            leaves += self.right_child.get_leaves_below()
        return leaves

    def update_bounds_below(self):
        """Method that updates the bounds of the node and its children"""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        for child in [self.left_child, self.right_child]:
            if child:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()

                if child is self.left_child:
                    child.lower[self.feature] = self.threshold
                else:
                    child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """Method that computes the indicator function for this node"""

        def is_large_enough(x):
            """Return True for individuals where
            feature > lower for ALL features"""
            return np.all(
                np.array(
                    [np.greater(
                        x[:, key], self.lower[key]) for key in list(
                            self.lower.keys())]), axis=0)

        def is_small_enough(x):
            """Return True for individuals where
            feature <= upper for ALL features"""
            return np.all(
                np.array(
                    [np.less_equal(
                        x[:, key], self.upper[key]) for key in list(
                            self.upper.keys())]), axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        """Method that predicts the value of x using the node"""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """Class Leaf that represents a leaf in a decision tree"""
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Method that returns the maximum depth of the tree below the leaf"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """A leaf always counts as 1"""
        return 1

    def __str__(self):
        """String representation of a leaf"""
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """Method that returns a list of the leaves below the node"""
        return [self]

    def update_bounds_below(self):
        """Method that updates the bounds of the leaf"""
        pass

    def pred(self, x):
        """Method that predicts the value of x using the leaf"""
        return self.value


class Decision_Tree():
    """Class Decision_Tree that represents a decision tree"""
    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Method that returns the maximum depth of the tree"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Method that returns the number of nodes in the tree"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Returns the string representation of the root node"""
        return self.root.__str__()

    def get_leaves(self):
        """Method that returns a list of the leaves in the tree"""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Method that updates the bounds of the tree"""
        self.root.update_bounds_below()

    def update_predict(self):
        """Method that computes the prediction function"""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        leaf_values = np.array([leaf.value for leaf in leaves])

        self.predict = lambda A: leaf_values[np.argmax(
            np.array([leaf.indicator(A) for leaf in leaves]), axis=0)]

    def pred(self, x):
        """Method that predicts the value of x using the decision tree"""
        return self.root.pred(x)

    def fit(self, explanatory, target, verbose=0):
        """Method that builds the decision tree by fitting it to the data"""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)

        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}
    - Accuracy on training data : {self.accuracy(
                                self.explanatory, self.target)}""")

    def np_extrema(self, arr):
        """Method that returns the minimum and maximum of a numpy array"""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Method that returns a random feature
        and threshold for splitting a node"""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x)*feature_min + x * feature_max
        return feature, threshold

    def fit_node(self, node):
        """Recursively build the tree by splitting nodes"""
        node.feature, node.threshold = self.split_criterion(node)

        # Diviser les individus Left / Right
        left_population = node.sub_population & (
             self.explanatory[:, node.feature] > node.threshold)
        right_population = node.sub_population & (
             self.explanatory[:, node.feature] <= node.threshold)

        # Vérifier si left est une feuille
        left_target_unique = len(np.unique(self.target[left_population]))
        is_left_leaf = (np.sum(left_population) < self.min_pop) or (
            node.depth + 1 == self.max_depth) or (
                left_target_unique == 1)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Vérifier si right est une feuille
        right_target_unique = len(np.unique(self.target[right_population]))
        is_right_leaf = (np.sum(right_population) < self.min_pop) or \
                        (node.depth + 1 == self.max_depth) or \
                        (right_target_unique == 1)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Method that returns a leaf child for a node"""
        value = np.bincount(self.target[sub_population]).argmax()
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Method that returns a node child for a node"""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """Method that computes the accuracy
        of the decision tree on test data"""
        return np.sum(
            np.equal(
                self.predict(
                    test_explanatory), test_target)) / test_target.size

    def possible_thresholds(self, node, feature):
        """Method that returns the possible thresholds for splitting a node"""
        values = np.unique((self.explanatory[:, feature])[node.sub_population])
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        """Find the best threshold for this feature using Gini impurity"""

        # Obtenir les seuils possibles
        thresholds = self.possible_thresholds(node, feature)

        # Extraire les données du nœud
        mask = node.sub_population
        feature_values = self.explanatory[mask, feature]  # (n,)
        target_values = self.target[mask]  # (n,)
        n_samples = len(feature_values)

        # Obtenir les classes uniques
        classes = np.unique(target_values)
        n_classes = len(classes)

        # Créer encodage one-hot : is_class[i, k] =
        # True si individu i est classe k
        is_class = target_values[:, np.newaxis] == classes  # (n, c)

        # Créer Left_F : goes_left[i, j] = True si feature[i] > threshold[j]
        goes_left = feature_values[:, np.newaxis] > thresholds  # (n, t)

        # Compter combien de chaque classe va à gauche : (t, c)
        left_counts = np.sum(
            goes_left[:, :, np.newaxis] & is_class[:, np.newaxis, :], axis=0)

        # Total à gauche et droit pour chaque seuil
        left_total = np.sum(left_counts, axis=1)  # (t,)
        right_total = n_samples - left_total  # (t,)

        # Total de chaque classe
        class_counts = np.sum(is_class, axis=0)  # (c,)
        right_counts = class_counts - left_counts  # (t, c)

        # Calculer Gini pour left children
        left_proportions = left_counts / left_total[:, np.newaxis]  # (t, c)
        left_gini = 1 - np.sum(left_proportions ** 2, axis=1)  # (t,)

        # Calculer Gini pour right children
        right_proportions = right_counts / right_total[:, np.newaxis]  # (t, c)
        right_gini = 1 - np.sum(right_proportions ** 2, axis=1)  # (t,)

        # Gini split pondéré
        gini_splits = (
            left_total * left_gini + right_total * right_gini) / n_samples

        # Trouver le meilleur seuil (minimum Gini)
        best_idx = np.argmin(gini_splits)
        best_threshold = thresholds[best_idx]
        best_gini = gini_splits[best_idx]

        return best_threshold, best_gini

    def Gini_split_criterion(self, node):
        """Method that returns the feature and
        threshold for splitting a node"""
        X = np.array([self.Gini_split_criterion_one_feature(node, i)
                      for i in range(self.explanatory.shape[1])])
        i = np.argmin(X[:, 1])
        return i, X[i, 0]
