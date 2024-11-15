import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from feature_extraction import get_resnet_feature_extractor, extract_features, reduce_features_with_pca
from data.download import load


class CustomDecisionTree:
    def __init__(self, max_depth=50):
        self.max_depth = max_depth
        self.tree = None

    def gini(self, y):
        """Compute Gini Impurity."""
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def split(self, X, y, index, value):
        """Split dataset based on feature index and value."""
        left = y[X[:, index] <= value]
        right = y[X[:, index] > value]
        return left, right

    def best_split(self, X, y):
        """Find the best split for the current dataset."""
        best_index, best_value, best_score = None, None, float('inf')
        for index in range(X.shape[1]):
            for value in np.unique(X[:, index]):
                left, right = self.split(X, y, index, value)
                gini_split = (len(left) / len(y)) * self.gini(left) + (len(right) / len(y)) * self.gini(right)
                if gini_split < best_score:
                    best_index, best_value, best_score = index, value, gini_split
        return best_index, best_value

    def build_tree(self, X, y, depth=0):
        """Recursively build the decision tree."""
        if len(np.unique(y)) == 1 or depth == self.max_depth:
            return np.bincount(y).argmax()
        index, value = self.best_split(X, y)
        left_indices = X[:, index] <= value
        right_indices = X[:, index] > value
        return {
            'index': index,
            'value': value,
            'left': self.build_tree(X[left_indices], y[left_indices], depth + 1),
            'right': self.build_tree(X[right_indices], y[right_indices], depth + 1)
        }

    def fit(self, X, y):
        """Fit the decision tree."""
        self.tree = self.build_tree(X, y)

    def predict_single(self, tree, x):
        """Predict the label for a single example."""
        if isinstance(tree, dict):
            if x[tree['index']] <= tree['value']:
                return self.predict_single(tree['left'], x)
            else:
                return self.predict_single(tree['right'], x)
        return tree

    def predict(self, X):
        """Predict labels for all examples."""
        return np.array([self.predict_single(self.tree, x) for x in X])


def run_decision_tree():
    # Load Data
    train_loader, test_loader = load(sample_per_class=500, batch_size=32)

    # Feature Extraction
    resnet = get_resnet_feature_extractor()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_features, train_labels = extract_features(resnet, train_loader, device)
    test_features, test_labels = extract_features(resnet, test_loader, device)

    # PCA
    train_features_reduced, pca_model = reduce_features_with_pca(train_features, n_components=50)
    test_features_reduced = pca_model.transform(test_features)

    # Custom Decision Tree
    custom_tree = CustomDecisionTree(max_depth=50)
    print("Training Custom Decision Tree...")
    custom_tree.fit(train_features_reduced, train_labels)
    print("Evaluating Custom Decision Tree...")
    y_pred_custom = custom_tree.predict(test_features_reduced)

    # Scikit-Learn Decision Tree
    sklearn_tree = DecisionTreeClassifier(max_depth=50, criterion='gini')
    print("Training Scikit-Learn Decision Tree...")
    sklearn_tree.fit(train_features_reduced, train_labels)
    print("Evaluating Scikit-Learn Decision Tree...")
    y_pred_sklearn = sklearn_tree.predict(test_features_reduced)

    # Evaluate Both Models
    results = {}
    for model_name, y_pred in [("Custom Tree", y_pred_custom), ("Scikit-Learn Tree", y_pred_sklearn)]:
        accuracy = accuracy_score(test_labels, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(test_labels, y_pred, average='weighted')
        conf_matrix = confusion_matrix(test_labels, y_pred)
        results[f"{model_name} Accuracy"] = accuracy
        results[f"{model_name} Precision"] = precision
        results[f"{model_name} Recall"] = recall
        results[f"{model_name} F1-Score"] = f1
        print(f"\n{model_name} Results:")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
        print(f"Confusion Matrix:\n{conf_matrix}")

    return results
