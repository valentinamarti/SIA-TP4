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
