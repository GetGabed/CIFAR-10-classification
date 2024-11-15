import time

import torch

from naive_bayes import run_naive_bayes
from decision_tree import run_decision_tree
from mlp import run_mlp
from cnn import run_cnn

def main():
    print("Starting Image Classification Project...")

    print(f"Using device: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "Using device: CPU")

    # Start the total timer
    total_start_time = time.time()

    # Run Naive Bayes
    print("\nRunning Naive Bayes...")
    start_time = time.time()
    naive_bayes_results = run_naive_bayes()
    end_time = time.time()
    naive_bayes_time = (end_time - start_time) / 60

    # Run Decision Tree
    print("\nRunning Decision Tree...")
    start_time = time.time()
    decision_tree_results = run_decision_tree()
    end_time = time.time()
    decision_tree_time = (end_time - start_time) / 60

    # Run Multi-Layer Perceptron
    print("\nRunning Multi-Layer Perceptron...")
    start_time = time.time()
    mlp_results = run_mlp()
    end_time = time.time()
    mlp_time = (end_time - start_time) / 60

    # Run Convolutional Neural Network
    print("\nRunning Convolutional Neural Network...")
    start_time = time.time()
    cnn_results = run_cnn()
    end_time = time.time()
    cnn_time = (end_time - start_time) / 60

    # End the total timer
    total_end_time = time.time()
    total_time = (total_end_time - total_start_time) / 60

    # Print or save results
    print("\n--- Model Comparisons ---")
    for model_name, metrics in [
        ("Naive Bayes", naive_bayes_results),
        ("Decision Tree", decision_tree_results),
        ("MLP", mlp_results),
        ("CNN", cnn_results),
    ]:
        print(f"\n{model_name} Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    print("\n--- Summary of Time Taken ---")
    print(f"Time Taken for Naive Bayes: {naive_bayes_time:.2f} minutes")
    print(f"Time Taken for Decision Tree: {decision_tree_time:.2f} minutes")
    print(f"Time Taken for MLP: {mlp_time:.2f} minutes")
    print(f"Time Taken for CNN: {cnn_time:.2f} minutes")
    print(f"Total Time Taken: {total_time:.2f} minutes")

if __name__ == "__main__":
    main()
