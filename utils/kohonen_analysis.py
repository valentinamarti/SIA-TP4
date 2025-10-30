import pandas as pd

from europe.kohonen_net import KohonenNet
from graphs.kohonen_graphs import plot_u_matrix, plot_hit_map, plot_component_plane, plot_all_component_planes, \
    plot_bmu_table
from parser.parser import load_and_preprocess_data


def run_kohonen_analysis(filepath, map_rows, map_cols, epochs, initial_eta, initial_radius, init_method='sample', output_name='kohonen_results'):
    """
    Performs the complete analysis for Exercise 1.1 of the Kohonen Network:
    Loads data, trains the network, calculates metrics (U-Matrix, Hit Map), and generates plots.

    :param filepath: Path to the CSV file.
    :param map_rows: Number of grid rows.
    :param map_cols: Number of grid columns.
    :param epochs: Number of training passes.
    :param initial_eta: Initial learning rate.
    :param initial_radius: Initial neighborhood radius.
    :param init_method: Weight initialization method ('random' [0,1] or 'sample').
    """
    print(f"\n--- Iniciando Análisis de Kohonen para {filepath} ({map_rows}x{map_cols}) ---")

    X_scaled, countries, feature_names = load_and_preprocess_data(filepath)
    INPUT_DIM = X_scaled.shape[1]

    print(f"Datos cargados. Features (N)={INPUT_DIM}. Países={len(countries)}.")

    net = KohonenNet(map_rows, map_cols, INPUT_DIM)
    final_weights = net.fit(X_scaled, epochs, initial_eta, initial_radius, init_method)

    print("\n--- Calculando Métricas de Análisis ---")
    count_map = net.calculate_hit_map(X_scaled)
    u_matrix = net.calculate_u_matrix()
    mapping_df = net.map_data_to_bmus(X_scaled, countries)


    print("\n[Tabla] Mapeo de Países a Neuronas (Asociación):")
    print(mapping_df.sort_values(by=['BMU_Row', 'BMU_Col']).to_string())

    plot_u_matrix(u_matrix, map_rows, map_cols, output_name)
    plot_hit_map(count_map, map_rows, map_cols, output_name)

    try:
        print("\n--- Generando Gráficos de Componentes para todas las variables ---")
        plot_all_component_planes(final_weights, feature_names, map_rows, map_cols, output_name)
    except NameError:
        print(
            "Advertencia: No se pudo generar el mapa de todas las componentes. ¿Se agregó la función plot_all_component_planes a kohonen_graphs.py y se importó correctamente?")

    try:
        plot_bmu_table(mapping_df, map_rows, map_cols, output_name)
    except NameError:
        print("Advertencia: No se pudo generar el PNG de la tabla. Revisar import y definición de plot_bmu_table.")

    print("\nAnálisis de Kohonen finalizado. Gráficos generados.")

    # Retorna los pesos finales y el mapeo para futuras referencias si es necesario
    return final_weights, mapping_df