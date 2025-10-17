import argparse

from graphs.graphs import plot_biplot, plot_bar_chart, plot_boxplot
from parser.parser import upload_data
from utils.utils import apply_pca, show_pc1_data


def main(filepath):
    """Main function to run the complete PCA workflow."""
    # 1. Load and prepare data
    print(f"Step 1: Loading and standardizing data from '{filepath}'...")
    scaled_data, country_names, feature_names = upload_data(filepath)
    print("Data ready.\n")

    # 2. Apply PCA
    print("Step 2: Applying Principal Component Analysis (PCA)...")
    pca_model, projected_data = apply_pca(scaled_data)
    print("PCA calculated successfully.\n")

    # 3. Generate plots
    print("Step 3: Generating plots...")
    plot_biplot(projected_data, pca_model, country_names, feature_names)
    print("- 'biplot.png' saved.")

    # Extract the scores of the first component
    pc1_scores = projected_data[:, 0]

    plot_bar_chart(pc1_scores, country_names)
    print("- 'pca_bar_index.png' saved.")

    plot_boxplot(scaled_data, feature_names)
    print("- 'pca_boxplot.png' saved.\n")

    # 4. Interpret the first principal component
    print("Step 4: Analyzing the First Principal Component (PC1)...")
    show_pc1_data(pca_model, feature_names)
    print("\nThis loadings vector is what you should compare with your Oja's rule implementation.")

    print("\n--- Analysis complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "filepath",
        type=str,
        help="La ruta al archivo CSV que contiene los datos (ej: europe.csv)"
    )

    args = parser.parse_args()

    main(args.filepath)
