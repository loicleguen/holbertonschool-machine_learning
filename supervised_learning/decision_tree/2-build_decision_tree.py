#!/usr/bin/env python3
"""Build a decision tree with the ID3 algorithm"""

import numpy as np


class Node:
    """Class that represents a node in a decision tree"""
    def __init__(self, feature=None,
                 threshold=None,
                 left_child=None,
                 right_child=None,
                 is_root=False,
                 depth=0):
        """Initialize a node"""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Return the maximum depth of the tree below this node"""
        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """Return the number of nodes below this node"""
        left = self.left_child.count_nodes_below(only_leaves=only_leaves)
        right = self.right_child.count_nodes_below(only_leaves=only_leaves)

        if only_leaves:
            return left + right
        else:
            return 1 + left + right

    def left_child_add_prefix(self, text):
        """Add a prefix to the left child of this node"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Add a prefix to the right child of this node"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("           " + x) + "\n"
        return new_text

    def __str__(self):
        """Return a string representation of the node"""
        if self.is_root:
            node_repr = (
                f"root [feature={self.feature}, threshold={self.threshold}]")
        else:
            node_repr = (
                f"-> node [feature={self.feature}, threshold={self.threshold}]"
                )

        left_str = self.left_child.__str__()
        right_str = self.right_child.__str__()

        result = node_repr + "\n"
        result += self.left_child_add_prefix(left_str)
        result += self.right_child_add_prefix(right_str)

        return result.rstrip("\n")


class Leaf(Node):
    """Class that represents a leaf in a decision tree"""
    def __init__(self, value, depth=None):
        """Initialize a leaf node"""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Return the maximum depth of the tree below this node"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Return the number of nodes below this node"""
        return 1

    def __str__(self):
        """Return a string representation of the leaf node"""
        return (f"-> leaf [value={self.value}]")


class Decision_Tree():
    """Class that represents a decision tree"""
    def __init__(self,
                 max_depth=10,
                 min_pop=1,
                 seed=0,
                 split_criterion="random",
                 root=None):
        """Initialize the decision tree"""
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
        """Return the depth of the tree"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Return the number of nodes in the tree"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Return a string representation of the tree"""
        return self.root.__str__()
