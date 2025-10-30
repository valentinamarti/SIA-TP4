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

    print("\n========================================================")
    print("        DEBUGGING: DATOS ESTANDARIZADOS (Z-SCORE)       ")
    print("========================================================")
    print("1. Estadísticos (Media y Desv. Estándar):")
    print(f"   Media: {media.to_string()}")
    print(f"   Desv. Estándar: {desviacion_estandar.to_string()}")
    print("-" * 50)

    # Extraer y mostrar el vector estandarizado de ESPAÑA
    try:
        spain_row = features_scaled[paises == 'Spain']
        germany_row = features_scaled[paises == 'Germany']

        print(f"2. Vector Estandarizado de ESPAÑA (Problema):")
        print(spain_row.to_string(float_format='%.4f'))
        print("-" * 50)

        print(f"3. Vector Estandarizado de ALEMANIA (Contraste):")
        print(germany_row.to_string(float_format='%.4f'))
        print("-" * 50)

    except:
        print("Advertencia: No se pudo encontrar España o Alemania en el DataFrame.")

    print("========================================================")

    # ----------------------------------------------

    return features_scaled_array, paises, feature_names