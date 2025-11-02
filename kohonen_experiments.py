import pandas as pd
import itertools
import os
import sys
import time

from utils.kohonen_analysis import run_kohonen_analysis


def run_kohonen_experiments(filepath: str, experiments_data: list):
    """
    Ejecuta múltiples experimentos de Kohonen y recopila los resultados.

    :param filepath: Ruta al archivo de datos (ej: europe.csv).
    :param experiments_data: Lista de diccionarios de hiperparámetros.
    :return: DataFrame con los resultados de cada corrida.
    """
    results = []

    for i, params in enumerate(experiments_data):
        experiment_name = f"exp_{i + 1}_{params['map_rows']}x{params['map_cols']}_eta{params['initial_eta']}"
        print(f"\n--- Ejecutando Experimento {i + 1}/{len(experiments_data)}: {experiment_name} ---")

        start_time = time.time()

        try:
            kohonen_args = {
                'map_rows': params['map_rows'],
                'map_cols': params['map_cols'],
                'epochs': params['epochs'],
                'initial_eta': params['initial_eta'],
                'initial_radius': params.get('initial_radius', 4.0),
                'eta_adaptive': params.get('eta_adaptive', True),
                'radius_adaptive': params.get('radius_adaptive', True),
                'init_method': params.get('init_method', 'sample'),
                'output_name': experiment_name
            }

            final_weights, mapping_df = run_kohonen_analysis(
                filepath,
                **kohonen_args
            )

        except Exception as e:
            print(f"Error en experimento {experiment_name}: {e}")
            q_error = None
            t_error = None

        end_time = time.time()


        results.append({
            'experiment_name': experiment_name,
            'map_rows': params['map_rows'],
            'map_cols': params['map_cols'],
            'map_size': params['map_rows'] * params['map_cols'],
            'epochs': params['epochs'],
            'initial_eta': params['initial_eta'],
            'initial_radius': params.get('initial_radius', 4.0),
            'eta_adaptive': params.get('eta_adaptive', True),
            'radius_adaptive': params.get('radius_adaptive', True),
            'init_method': params.get('init_method', 'sample'),
            'Execution_Time_s': round(end_time - start_time, 2)

        })

    return pd.DataFrame(results)


# 1. IMPACTO DEL TAMAÑO (3 Exp.): Mantenemos ETA y Radio fijos (control).
size_impact_experiments = list(itertools.product(
    [(2, 2), (3, 3), (4, 4)],  # MAP_SIZES (E1, E2, E3)
    [500],  # EPOCHS
    [0.5],  # ETAS
    [4.0]  # RADIUS
))

# 2. IMPACTO DE ETA (2 Exp.): Mantenemos Tamaño y Radio fijos (4x4, 4.0).
eta_impact_experiments = list(itertools.product(
    [(4, 4)],  # MAP_SIZE (Fijo)
    [500],  # EPOCHS
    [0.1, 0.7],  # ETAS (E4, E5)
    [4.0]  # RADIUS
))

# 3. IMPACTO DE RADIO (2 Exp.): Mantenemos Tamaño y ETA fijos (4x4, 0.5).
radio_impact_experiments = list(itertools.product(
    [(4, 4)],  # MAP_SIZE (Fijo)
    [500],  # EPOCHS
    [0.5],  # ETAS
    [2.0, 6.0]  # RADIUS (E6, E7)
))

# 4. LÍMITES Y SINERGIA (2 Exp.): Combina efectos extremos.
# Excluye combinaciones que ya están cubiertas (ej. 4x4, 0.5, 4.0 es E2)
synergy_experiments = list(itertools.product(
    [(4, 4)],  # MAP_SIZE (Fijo)
    [500],  # EPOCHS
    [0.1, 0.7],  # ETAS (E8 usa 0.1, E9 usa 0.7)
    [2.0, 4.0]  # RADIUS (E8 usa 2.0, E9 usa 4.0)
))

# Eliminar duplicados y aislar los 9 experimentos deseados:
all_combinations = []
all_combinations.extend(size_impact_experiments)
all_combinations.extend(eta_impact_experiments)
all_combinations.extend(radio_impact_experiments)
all_combinations.extend(synergy_experiments)

# Limpiar duplicados (el 4x4 base se genera en varias listas, pero solo necesitamos 9 únicos)
unique_combinations = []
seen = set()
for combo in all_combinations:
    # (rows, cols), epochs, eta, radius
    key = (combo[0][0], combo[0][1], combo[1], combo[2], combo[3])
    if key not in seen:
        unique_combinations.append(combo)
        seen.add(key)


experiments = []
for (rows, cols), epochs, eta, radius in unique_combinations:
    experiments.append({
        'map_rows': rows,
        'map_cols': cols,
        'epochs': epochs,
        'initial_eta': eta,
        'initial_radius': radius,
        'init_method': 'sample'
    })

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python run_experiments.py <ruta_al_archivo_csv>")
        sys.exit(1)

    DATA_FILEPATH = sys.argv[1]
    OUTPUT_CSV = f"kohonen_experiments_results_{time.strftime('%Y%m%d_%H%M%S')}.csv"

    OUTPUT_BASE_NAME = f"kohonen_experiments_results_{time.strftime('%Y%m%d_%H%M%S')}"

    print(f"Total de experimentos a ejecutar: {len(experiments)}")

    results_df = run_kohonen_experiments(DATA_FILEPATH, experiments)

    results_df.to_csv(f'./results/{OUTPUT_CSV}', index=False)
    print(f"\nTodos los experimentos han finalizado.")
    print(f"Resultados guardados en: ./results/{OUTPUT_CSV}")
    print("\nResumen de los mejores resultados (buscando menor Error de Cuantización):")
