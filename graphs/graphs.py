import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_biplot(projected_data, pca, country_labels, feature_labels, name):
    scores_x, scores_y = projected_data[:, 0], projected_data[:, 1]
    loadings_x, loadings_y = pca.components_[0, :], pca.components_[1, :]

    plt.figure(figsize=(14, 10))
    sns.set_style("whitegrid")
    sns.scatterplot(x=scores_x, y=scores_y, s=70, alpha=0.7)

    for i, label in enumerate(country_labels):
        plt.text(scores_x[i], scores_y[i], label, fontsize=9)

    scale_factor = 1.5 * np.max(np.abs(projected_data[:, :2]))
    for i, feature in enumerate(feature_labels):
        plt.arrow(0, 0, loadings_x[i] * scale_factor, loadings_y[i] * scale_factor,
                  color='r', alpha=0.8, head_width=0.1)
        plt.text(loadings_x[i] * scale_factor * 1.15, loadings_y[i] * scale_factor * 1.15,
                 feature, color='r', ha='center', va='center')

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    plt.title('Biplot of European Countries (PCA)')
    plt.grid(True)
    filename = f"results/{name.lower()}_biplot.png"
    plt.savefig(filename)
    plt.savefig('results/biplot.png')
    plt.close()

def plot_bar_chart(pc1_scores, countries, name):
    df_index = pd.DataFrame({'Country': countries, 'PC1_Score': pc1_scores})
    df_index_sorted = df_index.sort_values(by='PC1_Score', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Country', y='PC1_Score', data=df_index_sorted, palette='viridis')
    plt.xticks(rotation=90, fontsize=10)
    plt.title(name+' - Bar Chart: PC1 Index by Country')
    plt.ylabel('First Principal Component Score (PC1)')
    plt.xlabel('Country')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    filename = f"results/{name}_bar_index.png"
    plt.savefig(filename)
    plt.close()


def plot_boxplot(scaled_data, feature_names, name):
    df_scaled_features = pd.DataFrame(scaled_data, columns=feature_names)

    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df_scaled_features, palette='coolwarm')
    plt.title('Distribution of Standardized Features Across All Countries')
    plt.ylabel('Standardized Value (Z-score)')
    plt.xlabel('Feature')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    filename = f"results/{name.lower()}_boxplot.png"
    plt.savefig(filename)
    plt.close()