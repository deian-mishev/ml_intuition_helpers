import pandas as pd
import numpy as np
from data.data import generate_tree_viz


def entropy(p):
    if p == 0 or p == 1:
        return 0
    else:
        return -p * np.log2(p) - (1 - p)*np.log2(1 - p)


def information_gain(X, y, left_indices, right_indices):
    """
    Here, X has the elements in the node and y is theirs respectives classes
    """
    p_node = sum(y)/len(y)
    h_node = entropy(p_node)
    w_entropy = weighted_entropy(X, y, left_indices, right_indices)
    return h_node - w_entropy


def weighted_entropy(X, y, left_indices, right_indices):
    """
    This function takes the splitted dataset, the indices we chose to split and returns the weighted entropy.
    """
    w_left = len(left_indices)/len(X)
    w_right = len(right_indices)/len(X)
    p_left = sum(y[left_indices])/len(left_indices)
    p_right = sum(y[right_indices])/len(right_indices)

    weighted_entropy = w_left * entropy(p_left) + w_right * entropy(p_right)
    return weighted_entropy


def split_indices(X, index_feature):
    """
    Given a dataset and a index feature, return two lists for the two split nodes.
    """
    left_indices = []
    right_indices = []
    for i, x in enumerate(X):
        if x[index_feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices, right_indices


def compute_entropy(y):
    entropy = 0

    if len(y) == 0:
        return 0
    entropy = sum(y[y == 1])/len(y)
    if entropy == 0 or entropy == 1:
        return 0
    else:
        return -entropy*np.log2(entropy) - (1-entropy)*np.log2(1-entropy)


def split_dataset_continuous(X, node_indices, feature, threshold):
    left_indices = []
    right_indices = []
    for i in node_indices:
        if X[i][feature] <= threshold:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices, right_indices


def get_potential_splits(X, node_indices, feature):
    values = sorted(set([X[i][feature] for i in node_indices]))
    thresholds = [(values[i] + values[i+1])/2 for i in range(len(values)-1)]
    return thresholds


def split_dataset(X, node_indices, feature):
    left_indices = []
    right_indices = []

    for i in node_indices:
        if X[i][feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)

    return left_indices, right_indices


def compute_information_gain(X, y, node_indices, feature):

    left_indices, right_indices = split_dataset(X, node_indices, feature)

    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]

    information_gain = 0

    node_entropy = compute_entropy(y_node)
    left_entropy = compute_entropy(y_left)
    right_entropy = compute_entropy(y_right)
    w_left = len(X_left) / len(X_node)
    w_right = len(X_right) / len(X_node)
    weighted_entropy = w_left * left_entropy + w_right * right_entropy
    information_gain = node_entropy - weighted_entropy

    return information_gain


def get_best_split(X, y, node_indices):
    num_features = X.shape[1]
    best_feature = -1
    best_threshold = None
    max_info_gain = -float("inf")

    for feature in range(num_features):
        thresholds = get_potential_splits(X, node_indices, feature)

        for threshold in thresholds:
            left_indices, right_indices = split_dataset_continuous(X, node_indices, feature, threshold)
            if len(left_indices) == 0 or len(right_indices) == 0:
                continue

            X_node = X[node_indices]
            y_node = y[node_indices]
            y_left = y[left_indices]
            y_right = y[right_indices]

            node_entropy = compute_entropy(y_node)
            left_entropy = compute_entropy(y_left)
            right_entropy = compute_entropy(y_right)

            w_left = len(y_left) / len(y_node)
            w_right = len(y_right) / len(y_node)
            weighted_entropy = w_left * left_entropy + w_right * right_entropy
            info_gain = node_entropy - weighted_entropy

            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold


def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth, tree):
    if current_depth == max_depth or len(set(y[node_indices])) == 1:
        formatting = " " * current_depth + "-" * current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return

    best_feature, best_threshold = get_best_split(X, y, node_indices)
    formatting = "-" * current_depth
    print(f"{formatting} Depth {current_depth}, {branch_name}: Split on feature {best_feature} at threshold {best_threshold:.2f}")

    left_indices, right_indices = split_dataset_continuous(X, node_indices, best_feature, best_threshold)
    tree.append((left_indices, right_indices, best_feature, best_threshold))

    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth + 1, tree)
    build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth + 1, tree)
    return tree


if __name__ == "__main__":
    X_train = np.array([[1, 1, 1, 7.8],
                        [0, 0, 1, 8.1],
                        [0, 1, 0, 9.3],
                        [1, 0, 1, 8.9],
                        [1, 1, 1, 8.3],
                        [1, 1, 0, 7.5],
                        [0, 0, 0, 9.1],
                        [1, 1, 0, 8.3],
                        [0, 1, 0, 7.7],
                        [0, 1, 0, 10.0]])

    y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])
    tree = []
    build_tree_recursive(X_train, y_train, [
                         0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "Root", max_depth=4, current_depth=0, tree=tree)
    generate_tree_viz([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], y_train, tree)
