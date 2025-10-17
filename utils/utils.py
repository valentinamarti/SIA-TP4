import pandas as pd
from sklearn.decomposition import PCA


def apply_pca(datos):
    pca = PCA()
    datos_pca = pca.fit_transform(datos)
    return pca, datos_pca


def show_pc1_data(pca, feature_names):
    pc1_loadings = pca.components_[0]
    loadings_series = pd.Series(pc1_loadings, index=feature_names).sort_values(ascending=False)

    print("--- Composition of the First Principal Component (PC1) ---")
    print(loadings_series)
    print("\n--- Interpretation of PC1 ---")
    print("PC1 represents an axis of 'economic prosperity and quality of life'.")

    return loadings_series