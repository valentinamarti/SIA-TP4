import numpy as np
import pandas as pd

from graphs.graphs import plot_bar_chart
from parser.parser import upload_data
from utils.utils import apply_pca


def ojas_rule(X: np.ndarray,
              learning_rate: float = 1e-4,
              epochs: int = 200) -> np.ndarray:
    """
    Implements Oja's Rule to find the first principal component (PC1).

    Args:
        X (np.array): Standardized input data (mean 0).
        learning_rate (float): The learning rate (eta).
        epochs (int): The number of training epochs.

    Returns:
        np.array: The converged weight vector 'w' (the PC1).
    """
    n_samples, n_features = X.shape

    # Initialize Weights with uniform distribution between 0 and 1
    np.random.seed(42)
    w = np.random.rand(n_features)
    w = w / np.linalg.norm(w)  # Normalize to a unit vector

    print(f"Starting Oja's Rule... (Features: {n_features}, Epochs: {epochs}, LR: {learning_rate})")

    # Train
    for epoch in range(epochs):
        # stochastic learning iterating over each sample
        # x is a (n_features,) vector
        for x in X:

            # neuron output
            O = np.dot(w, x) # dot product of two arrays

            # weight change
            # Δw = η * (O*x - O²*w)
            delta_w = learning_rate * O * (x - O * w)

            # update weights
            w = w + delta_w

        # normalize the weight vector after EACH epoch
        # ensures 'w' converges to a unit eigenvector.
        w_norm = np.linalg.norm(w)
        if w_norm > 0:
            w = w / w_norm

    return w


def run_oja_analysis(filepath: str) -> None:
    """
    Main function to run the complete analysis for Exercise 1.2.
    1. Loads data
    2. Runs Oja's Rule implementation.
    3. Runs library-based PCA for comparison.
    4. Displays the comparison.
    """
    print(f"Step 1: Loading and standardizing data from '{filepath}'...")
    scaled_data, country_names, feature_names = upload_data(filepath)
    if scaled_data is None:
        return
    print("Data ready.\n")

    print("Running Oja's Rule to find PC1...")
    w_oja = ojas_rule(scaled_data, learning_rate=1e-4, epochs=1000)
    print("PC1 vector (Oja) calculated.\n")

    print("\n--- Calculating Scores (Projections) ---")
    scores_oja = scaled_data.dot(w_oja)

    scores_df = pd.DataFrame({'Country': country_names, 'PC1_Oja_Score': scores_oja})
    print("Scores calculated using Oja's vector:")
    print(scores_df.sort_values(by='PC1_Oja_Score', ascending=False).to_string())

    print("\n--- Generating Oja Bar Chart ---")
    chart_name = "oja"
    plot_bar_chart(scores_oja, country_names, name=chart_name)

    print("\nAnalysis complete. Check the 'results' folder for both '.png' charts.")