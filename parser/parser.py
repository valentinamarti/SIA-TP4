import pandas as pd
from sklearn.preprocessing import StandardScaler


def upload_data(filepath):

    df = pd.read_csv(filepath)

    paises = df['Country']
    features = df.drop('Country', axis=1)
    feature_names = features.columns.tolist()

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, paises, feature_names

def load_and_preprocess_data(filepath):
    """
    Loads the europe.csv file, separates the 'Country' column from the features,
    and manually standardizes the numerical features (Z-score: (x - mean) / standard deviation).

    Returns:
    - features_scaled_array: NumPy array of standardized features.
    - paises: Pandas Series containing the country names.
    - feature_names: List of strings with the feature names.
    """
    df = pd.read_csv(filepath)

    paises = df['Country']
    features = df.drop('Country', axis=1)
    feature_names = features.columns.tolist()

    # IMPLEMENTACIÓN MANUAL DE LA ESTANDARIZACIÓN Z-SCORE
    media = features.mean(axis=0)
    desviacion_estandar = features.std(axis=0, ddof=0)


    features_scaled = (features - media) / desviacion_estandar
    features_scaled_array = features_scaled.values

    return features_scaled_array, paises, feature_names
