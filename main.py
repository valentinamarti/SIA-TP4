import argparse
import os
import sys

from europe.oja import run_oja_analysis
from europe.pca import pca
from patterns.patterns3 import run_a
from utils.kohonen_analysis import run_kohonen_analysis


def execute_pca_analysis(filepath: str) -> None:
    """Exercise  mandatory PCA"""
    print("--- Comando: pca ---")
    pca(filepath)
    print("Análisis de PCA finalizado.")


def execute_kohonen_analysis(filepath):
    """Exercise 1.1"""
    kohonen_params = {
        'map_rows': 12,
        'map_cols': 12,
        'epochs': 400,
        'initial_eta': 0.5,
        'initial_radius': 6.0,
        'init_method': 'sample'
    }
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

    parser_kohonen = subparsers.add_parser("kohonen", help="Ejecuta el análisis de la Red de Kohonen (Ejercicio 1.1).")
    parser_kohonen.add_argument(
        "filepath",
        type=str,
        help="La ruta al archivo CSV (ej: europe.csv)"
    )

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
    args = parser.parse_args()

    # if not os.path.exists(args.filepath):
    #     print(f"Error: El archivo no se encontró en la ruta: {args.filepath}")
    #     sys.exit(1)

    if args.command == "kohonen":
        execute_kohonen_analysis(args.filepath)
    elif args.command == "pca":
        execute_pca_analysis(args.filepath)
    elif args.command == "oja":
        execute_oja_analysis(args.filepath)
    elif args.command == "hopfield":
        run_a(p_noise=args.noise)