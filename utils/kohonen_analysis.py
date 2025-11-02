import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import time


from europe.kohonen_net import KohonenNet
from graphs.kohonen_graphs import plot_u_matrix, plot_hit_map, plot_component_plane, plot_all_component_planes, \
    plot_bmu_table
from parser.parser import load_and_preprocess_data



def run_kohonen_analysis(filepath, map_rows, map_cols, epochs, initial_eta, initial_radius, eta_adaptive=True, radius_adaptive=True, init_method='sample', output_name='kohonen_results'):
    """
    Performs the complete analysis for Exercise 1.1 of the Kohonen Network:
    Loads data, trains the network, calculates metrics (U-Matrix, Hit Map), and generates plots.

    :param filepath: Path to the CSV file.
    :param map_rows: Number of grid rows.
    :param map_cols: Number of grid columns.
    :param epochs: Number of training passes.
    :param initial_eta: Initial learning rate.
    :param initial_radius: Initial neighborhood radius.
    :param radius_adaptive: Value that indicates if the radius varies overtime.
    :param eta_adaptive: Value that indicates if the eta varies overtime.
    :param init_method: Weight initialization method ('random' [0,1] or 'sample').
    """
    print(f"\n--- Iniciando Análisis de Kohonen para {filepath} ({map_rows}x{map_cols}) ---")

    X_scaled, countries, feature_names = load_and_preprocess_data(filepath)
    INPUT_DIM = X_scaled.shape[1]

    print(f"Datos cargados. Features (N)={INPUT_DIM}. Países={len(countries)}.")

    net = KohonenNet(map_rows, map_cols, INPUT_DIM)
    final_weights = net.fit(X_scaled, epochs, initial_eta, initial_radius, init_method, eta_adaptive, radius_adaptive)

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

def run_kohonen_analysis_experiments(filepath, experiments, output_prefix="kohonen_experiments", plot_experiments=None):
    """
    Ejecuta múltiples experimentos de Kohonen desde cero (sin llamar a run_kohonen_analysis),
    calculando QE y TE, generando mapas y gráficos comparativos.

    :param filepath: Ruta al archivo CSV (ej: 'data/europe.csv')
    :param experiments: Lista de diccionarios de parámetros.
    :param output_prefix: Prefijo base para guardar resultados y gráficos.
    :param plot_experiments: Lista de índices de experimentos (1-based) para incluir en gráficos comparativos. 
                             Si es None, incluye todos los experimentos.
    :return: DataFrame con los resultados (QE, TE, tiempos, etc.)
    """

    if not os.path.exists("./results"):
        os.makedirs("./results")

    print(f"\n🚀 Ejecutando {len(experiments)} experimentos de Kohonen...\n")

    # === Cargar y preparar los datos ===
    X_scaled, countries, feature_names = load_and_preprocess_data(filepath)
    INPUT_DIM = X_scaled.shape[1]
    print(f"Datos cargados. Países={len(countries)}, Features={INPUT_DIM}")

    all_results = []

    for i, params in enumerate(experiments, start=1):
        exp_name = f"exp_{i}_{params['map_rows']}x{params['map_cols']}_eta{params['initial_eta']}_{params['init_method']}"
        print(f"\n🔹 [{i}/{len(experiments)}] Ejecutando experimento: {exp_name}")

        start_time = time.time()

        # --- 1️⃣ Inicializar y entrenar red ---
        net = KohonenNet(params['map_rows'], params['map_cols'], INPUT_DIM)
        final_weights = net.fit(
            X=X_scaled,
            epochs=params['epochs'],
            initial_eta=params['initial_eta'],
            initial_radius=params['initial_radius'],
            init_method=params.get('init_method', 'sample'),
            eta_adaptive=params.get('eta_adaptive', True),
            radius_adaptive=params.get('radius_adaptive', True)
        )

        # --- 2️⃣ Calcular métricas individuales ---
        count_map = net.calculate_hit_map(X_scaled)
        u_matrix = net.calculate_u_matrix()
        mapping_df = net.map_data_to_bmus(X_scaled, countries)
        qe = net.calculate_quantization_error(X_scaled)
        te = net.calculate_topographic_error(X_scaled)

        duration = round(time.time() - start_time, 2)

        print(f"  ✓ QE={qe:.4f}, TE={te:.4f}, tiempo={duration}s")

        # --- 3️⃣ Guardar gráficos individuales ---
        plot_u_matrix(u_matrix, params['map_rows'], params['map_cols'], exp_name)
        plot_hit_map(count_map, params['map_rows'], params['map_cols'], exp_name)
        plot_all_component_planes(final_weights, feature_names, params['map_rows'], params['map_cols'], exp_name)
        plot_bmu_table(mapping_df, params['map_rows'], params['map_cols'], exp_name)

        # --- 4️⃣ Guardar resultados numéricos ---
        all_results.append({
            'Experiment': exp_name,
            'Rows': params['map_rows'],
            'Cols': params['map_cols'],
            'MapSize': params['map_rows'] * params['map_cols'],
            'Epochs': params['epochs'],
            'Initial_eta': params['initial_eta'],
            'Initial_radius': params['initial_radius'],
            'Eta_adaptive': params.get('eta_adaptive', True),
            'Radius_adaptive': params.get('radius_adaptive', True),
            'Init_method': params.get('init_method', 'sample'),
            'QE': round(qe, 4),
            'TE': round(te, 4),
            'Execution_s': duration
        })

    # --- 5️⃣ Consolidar y guardar resumen ---
    results_df = pd.DataFrame(all_results)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    csv_path = f"./results/{output_prefix}_results_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ Todos los experimentos completados. Resultados guardados en {csv_path}")

    # =====================================================
    # === 6️⃣ GRÁFICOS COMPARATIVOS AUTOMÁTICOS (QE, TE) ===
    # =====================================================
    
    # Filtrar resultados para gráficos comparativos si se especifica
    plot_df = results_df.copy()
    if plot_experiments is not None:
        # Filtrar por experimentos específicos (los nombres de experimento tienen el formato exp_N_...)
        plot_df = results_df[results_df['Experiment'].str.extract(r'exp_(\d+)_')[0].astype(int).isin(plot_experiments)]
        if plot_df.empty:
            print(f"\n⚠️  Advertencia: No se encontraron experimentos con índices {plot_experiments}. Generando gráficos para todos los experimentos.")
            plot_df = results_df.copy()
    
    if not plot_df.empty:
        sns.set(style="whitegrid", context="talk")

        # --- QE Barplot ---
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=plot_df.sort_values(by="QE", ascending=True),
            x="Experiment", y="QE", palette="Blues_r"
        )
        plt.xticks(rotation=45, ha="right")
        plt.title("Quantization Error (QE) por Experimento")
        plt.xlabel("Experimento")
        plt.ylabel("Quantization Error (QE)")
        plt.tight_layout()
        plt.savefig(f"./results/{output_prefix}_QE_barplot.png", bbox_inches="tight")
        plt.close()

        # --- TE Barplot ---
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=plot_df.sort_values(by="TE", ascending=True),
            x="Experiment", y="TE", palette="Reds_r"
        )
        plt.xticks(rotation=45, ha="right")
        plt.title("Topographic Error (TE) por Experimento")
        plt.xlabel("Experimento")
        plt.ylabel("Topographic Error (TE)")
        plt.tight_layout()
        plt.savefig(f"./results/{output_prefix}_TE_barplot.png", bbox_inches="tight")
        plt.close()

        print("\n📊 Gráficos comparativos generados en ./results/:")
        print(f"  - {output_prefix}_QE_barplot.png")
        print(f"  - {output_prefix}_TE_barplot.png")

    return results_df