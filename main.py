import argparse
import os
import sys

from europe.oja import run_oja_analysis
from europe.pca import pca
from patterns.patterns import run_a, run_b
from utils.kohonen_analysis import run_kohonen_analysis


def execute_pca_analysis(filepath: str) -> None:
    """Exercise  mandatory PCA"""
    print("--- Comando: pca ---")
    pca(filepath)
    print("Análisis de PCA finalizado.")


def execute_kohonen_analysis(
        filepath: str,
        map_rows: int,
        map_cols: int,
        epochs: int,
        initial_eta: float,
        initial_radius: float,
        init_method: str,
        output_name: str
):
    """Exercise 1.1 - Ejecuta el análisis de la Red de Kohonen con parámetros variables."""
    print(f"--- Comando: kohonen (Nombre de salida: {output_name}) ---")

    # Se crea el diccionario de parámetros a partir de los argumentos recibidos
    kohonen_params = {
        'map_rows': map_rows,
        'map_cols': map_cols,
        'epochs': epochs,
        'initial_eta': initial_eta,
        'initial_radius': initial_radius,
        'init_method': init_method,
        'output_name': output_name
    }
    print(f"Parámetros de Kohonen: {kohonen_params}")

    final_weights, mapping_df = run_kohonen_analysis(
        filepath,
        **kohonen_params
    )

    print("Análisis de Kohonen finalizado.")

def execute_oja_analysis(filepath: str) -> None:
    run_oja_analysis(filepath)
    print("Análisis de Oja finalizado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecuta diferentes análisis de SIA (Kohonen/PCA) en datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="El comando a ejecutar (kohonen o pca).")

    parser_pca = subparsers.add_parser("pca", help="Ejecuta el análisis de la Regla de Oja (PCA con librería) (Ejercicio 1.2).")
    parser_pca.add_argument(
        "filepath",
        type=str,
        help="La ruta al archivo CSV (ej: europe.csv)"
    )

    parser_oja = subparsers.add_parser("oja", help="Ejecuta el análisis de la Regla de Oja (Ejercicio 1.2).")
    parser_oja.add_argument(
        "filepath",
        type=str,
        help="Path to the CSV file (e.g., europe.csv)"
    )

    parser_hopfield = subparsers.add_parser("hopfield", help="Ejecuta Hopfield (Ejercicio 2.1).")
    parser_hopfield.add_argument(
        "noise",
        type=float,
        help="Noise to apply to original patterns"
    )
    parser_hopfield.add_argument(
        "exercise",
        type=str,
        help="Exercise to run (a or b)"
    )

    parser_kohonen = subparsers.add_parser("kohonen", help="Ejecuta el análisis de la Red de Kohonen (Ejercicio 1.1).")
    parser_kohonen.add_argument(
        "filepath",
        type=str,
        help="La ruta al archivo CSV (ej: europe.csv)"
    )

    # Nuevos argumentos de Kohonen con valores por defecto
    parser_kohonen.add_argument(
        "--rows",
        type=int,
        default=2,
        help="Cantidad de filas del mapa de Kohonen (map_rows). Default: 2."
    )
    parser_kohonen.add_argument(
        "--cols",
        type=int,
        default=2,
        help="Cantidad de columnas del mapa de Kohonen (map_cols). Default: 2."
    )
    parser_kohonen.add_argument(
        "--epochs",
        type=int,
        default=5000,
        help="Cantidad de épocas de entrenamiento. Default: 5000."
    )
    parser_kohonen.add_argument(
        "--eta",
        type=float,
        default=0.5,
        help="Tasa de aprendizaje inicial (initial_eta). Default: 0.5."
    )
    parser_kohonen.add_argument(
        "--radius",
        type=float,
        default=4.0,
        help="Radio inicial de vecindario (initial_radius). Default: 4.0."
    )
    parser_kohonen.add_argument(
        "--init-method",
        type=str,
        default='sample',
        choices=['random', 'sample'],
        help="Método de inicialización de pesos ('random' o 'sample'). Default: 'sample'."
    )
    parser_kohonen.add_argument(
        "--output-name",
        type=str,
        default='kohonen_results',
        help="Nombre base para los archivos de salida generados (ej: 'test_config_1'). Default: 'kohonen_results'."
    )

    args = parser.parse_args()

    # if not os.path.exists(args.filepath):
    #     print(f"Error: El archivo no se encontró en la ruta: {args.filepath}")
    #     sys.exit(1)

    if args.command == "kohonen":
        execute_kohonen_analysis(
            args.filepath,
            map_rows=args.rows,
            map_cols=args.cols,
            epochs=args.epochs,
            initial_eta=args.eta,
            initial_radius=args.radius,
            init_method=args.init_method,
            output_name=args.output_name
        )
    elif args.command == "pca":
        execute_pca_analysis(args.filepath)
    elif args.command == "oja":
        execute_oja_analysis(args.filepath)
    elif args.command == "hopfield":
        if args.exercise == "a":
            run_a(p_noise=args.noise)
        elif args.exercise == "b":
            run_b(p_noise=args.noise)
        else:
            print("Error: Ejercicio no válido. Use 'a' o 'b'.")
            sys.exit(1)