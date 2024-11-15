from naive_bayes import run_naive_bayes
# from decision_tree import run_decision_tree
# from mlp import run_mlp
# from cnn import run_cnn

def main():
    print("Starting Image Classification Project...")

    # Run Naive Bayes
    print("\nRunning Naive Bayes...")
    naive_bayes_results = run_naive_bayes()

    # Run Decision Tree
    print("\nRunning Decision Tree...")
    # decision_tree_results = run_decision_tree()

    # Run Multi-Layer Perceptron
    print("\nRunning Multi-Layer Perceptron...")
    # mlp_results = run_mlp()

    # Run Convolutional Neural Network
    print("\nRunning Convolutional Neural Network...")
    # cnn_results = run_cnn()

    # Print or save results
    print("\n--- Model Comparisons ---")
    for model_name, metrics in [
        ("Naive Bayes", naive_bayes_results),
        # ("Decision Tree", decision_tree_results),
        # ("MLP", mlp_results),
        # ("CNN", cnn_results),
    ]:
        print(f"\n{model_name} Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
