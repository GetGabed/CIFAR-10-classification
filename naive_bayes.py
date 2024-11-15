import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB

from data.download import load
from feature_extraction import get_resnet_feature_extractor, extract_features, reduce_features_with_pca


def run_naive_bayes():
    # Load Data
    train_loader, test_loader = load(sample_per_class=500, batch_size=32)

    # Extract Features
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet = get_resnet_feature_extractor()

    train_features, train_labels = extract_features(resnet, train_loader, device)
    test_features, test_labels = extract_features(resnet, test_loader, device)

    # Apply PCA
    train_features_reduced, pca_model = reduce_features_with_pca(train_features, n_components=50)
    test_features_reduced = pca_model.transform(test_features)

    # Custom Naive Bayes
    class GaussianNaiveBayes:
        def fit(self, X, y):
            self.classes = np.unique(y)
            self.mean = {cls: X[y == cls].mean(axis=0) for cls in self.classes}
            self.var = {cls: X[y == cls].var(axis=0) for cls in self.classes}
            self.priors = {cls: len(X[y == cls]) / len(y) for cls in self.classes}

        def predict(self, X):
            likelihoods = []
            for cls in self.classes:
                mean, var = self.mean[cls], self.var[cls]
                prior = self.priors[cls]
                likelihood = -0.5 * np.sum(np.log(2 * np.pi * var)) - 0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
                likelihood += np.log(prior)
                likelihoods.append(likelihood)
            return self.classes[np.argmax(likelihoods, axis=0)]

    custom_gnb = GaussianNaiveBayes()
    custom_gnb.fit(train_features_reduced, train_labels)
    y_pred_custom = custom_gnb.predict(test_features_reduced)

    # Scikit-Learn Naive Bayes
    sklearn_gnb = GaussianNB()
    sklearn_gnb.fit(train_features_reduced, train_labels)
    y_pred_sklearn = sklearn_gnb.predict(test_features_reduced)

    # Evaluate Both Models
    results = {}
    for model_name, y_pred in [("Custom GNB", y_pred_custom), ("Scikit-Learn GNB", y_pred_sklearn)]:
        accuracy = accuracy_score(test_labels, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(test_labels, y_pred, average='weighted')
        results[f"{model_name} Accuracy"] = accuracy
        results[f"{model_name} Precision"] = precision
        results[f"{model_name} Recall"] = recall
        results[f"{model_name} F1-Score"] = f1

    return results
