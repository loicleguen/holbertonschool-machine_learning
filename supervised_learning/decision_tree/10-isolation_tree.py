#!/usr/bin/env python3

import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    def __init__(self, max_depth=10, seed=0, root=None):
        """Class that represents an isolation tree"""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
            self.explanatory = None
            self.max_depth = max_depth
            self.predict = None
            self.min_pop = 1

    def __str__(self):
        """Returns the string representation of the root node"""
        return self.root.__str__()

    def depth(self):
        """Method that returns the maximum depth of the tree"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Method that returns the number of nodes in the tree"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """Method that updates the bounds of the tree"""
        self.root.update_bounds_below()

    def get_leaves(self):
        """Method that returns a list of the leaves in the tree"""
        return self.root.get_leaves_below()

    def update_predict(self):
        """Method that computes the prediction function"""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        leaf_values = np.array([leaf.value for leaf in leaves])

        self.predict = lambda A: leaf_values[np.argmax(
            np.array([leaf.indicator(A) for leaf in leaves]), axis=0)]

    def np_extrema(self, arr):
        """Method that returns the minimum and maximum of a numpy array"""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Choose a random feature and threshold"""
        diff = 0
        attempts = 0
        max_attempts = 10  # Limite pour éviter boucle infinie

        while diff == 0 and attempts < max_attempts:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
            attempts += 1

        # Si on n'a pas trouvé de feature avec variation,
        # retourner quand même quelque chose
        if diff == 0:
            threshold = feature_min
        else:
            x = self.rng.uniform()
            threshold = (1 - x) * feature_min + x * feature_max

        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """Method that returns a leaf child for a node"""
        leaf_child = Leaf(node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Method that returns a node child for a node"""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """Method that fits a node of the tree"""
        node.feature, node.threshold = self.random_split_criterion(node)

        left_population = node.sub_population & (
             self.explanatory[:, node.feature] > node.threshold)
        right_population = node.sub_population & (
             self.explanatory[:, node.feature] <= node.threshold)

        # Is left node a leaf ?
        is_left_leaf = (np.sum(left_population) < self.min_pop) or (
            node.depth + 1 == self.max_depth)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf ?
        is_right_leaf = (np.sum(right_population) < self.min_pop) or (
            node.depth + 1 >= self.max_depth)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """Method that builds the isolation tree"""
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones(
            explanatory.shape[0], dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")
