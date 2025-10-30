import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional
import seaborn as sns
import pandas as pd

def _get_staggered_coords(rows, cols, radius_data_units=0.45):
    """
    Calcula las coordenadas (x, y) en el plano para una cuadrícula escalonada (hexagonal).
    """
    x_coords = []
    y_coords = []

    # Distancia centro a centro.
    spacing_x = 2 * radius_data_units
    spacing_y = np.sqrt(3) * radius_data_units

    for r in range(rows):
        for c in range(cols):
            x = c * spacing_x + (radius_data_units * (r % 2))
            y = r * spacing_y

            x_coords.append(x)
            y_coords.append(y)

    # Solo devolvemos las coordenadas (x, y)
    return np.array(x_coords), np.array(y_coords)


def plot_circle_map(map_values, rows, cols, title, label, cmap, plot_type: str, filename: Optional[str] = None):
    """
    Genera un mapa compacto de círculos para una matriz 6x6, con calibración simple.
    """
    circle_radius = 1

    x_coords, y_coords = _get_staggered_coords(rows, cols, radius_data_units=circle_radius)

    # Ensure values are a flat array matching the coordinate order (row-major)
    map_values = np.asarray(map_values)
    if map_values.ndim == 2:
        map_values = map_values.reshape(rows * cols)
    elif map_values.ndim == 1 and map_values.size == rows * cols:
        pass
    else:
        map_values = map_values.reshape(rows * cols)

    fig_width = 10.0
    fig_height = 8.5
    plt.figure(figsize=(fig_width, fig_height))
    ax = plt.gca()

    norm = plt.Normalize(map_values.min(), map_values.max())

    marker_size = 2000

    plt.scatter(x_coords, y_coords,
                s=marker_size,
                c=map_values,
                cmap=cmap,
                marker='o',
                edgecolor='black',
                linewidths=0.5,
                norm=norm)

    plt.colorbar(label=label, ax=ax)
    plt.title(title, fontsize=14)


    margin = circle_radius * 1.2
    x_min = x_coords.min() - margin
    x_max = x_coords.max() + margin
    y_min = y_coords.min() - margin
    y_max = y_coords.max() + margin

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    if filename:
        results_dir = './results'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        save_path = os.path.join(results_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_u_matrix(u_matrix, rows, cols, output_name):
    """Generates the U-Matrix plot (compact circles) and saves it."""
    filename = f'u_matrix_{output_name}.png'
    plot_circle_map(u_matrix, rows, cols,
                    title='Unified Distance Matrix (U-Matrix) - Compact Topology',  # Título simplificado
                    label='Average Euclidean Distance (Dissimilarity)',
                    cmap='plasma',
                    plot_type='u_matrix',
                    filename=filename)


def plot_hit_map(count_map, rows, cols, output_name):
    """Generates the Count Map plot (compact circles) and saves it."""
    filename = f'hit_map_{output_name}.png'
    plot_circle_map(count_map, rows, cols,
                    title='Count Map (Hit Frequency) - Compact Topology',  # Título simplificado
                    label='Number of Associated Countries (Density)',
                    cmap='coolwarm',
                    plot_type='hit_map',
                    filename=filename)


def plot_component_plane(weights, feature_index, feature_name, rows, cols, output_name):
    """Generates the Component Plane plot (compact circles) for a single feature and saves it."""
    component_plane = weights[:, feature_index]

    safe_feature_name = feature_name.replace('.', '_').replace(' ', '_')
    filename = f'component_{safe_feature_name}_{output_name}.png'

    plot_circle_map(component_plane, rows, cols,
                    title=f'Component Map: Influence of "{feature_name}"',  # Título simplificado
                    label=f'Standardized Weight (Z-Score) of {feature_name}',
                    cmap='coolwarm',
                    plot_type='component',
                    filename=filename)


def plot_all_component_planes(weights, feature_names, map_rows, map_cols, output_name):
    """
    Genera y guarda un gráfico multi-panel de círculos escalonados
    (Component Planes) para todas las variables de entrada.

    :param weights: La matriz de pesos final. Se espera (N_neuronas x INPUT_DIM) o (map_rows, map_cols, INPUT_DIM).
    :param feature_names: Lista de nombres de las características.
    :param map_rows: Número de filas de la grilla.
    :param map_cols: Número de columnas de la grilla.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    circle_radius = 0.8
    marker_size = 1000
    cmap = 'coolwarm'


    if weights.ndim == 3:
        weights_2d = weights.reshape(map_rows * map_cols, weights.shape[2])
    elif weights.ndim == 2:
        weights_2d = weights
    else:
        print("Error: La matriz de pesos no tiene la dimensión esperada (2D o 3D).")
        return

    num_features = weights_2d.shape[1]

    x_coords, y_coords = _get_staggered_coords(map_rows, map_cols, radius_data_units=circle_radius)

    cols_layout = 3
    rows_layout = int(np.ceil(num_features / cols_layout))

    fig, axes = plt.subplots(rows_layout, cols_layout, figsize=(5 * cols_layout, 5 * rows_layout))
    axes = axes.flatten()  # Asegura que sea un array 1D para iterar

    for i in range(num_features):
        ax = axes[i]

        component_plane_values = weights_2d[:, i]

        # Normalización local (por feature) para maximizar el contraste
        vmax = np.abs(component_plane_values).max()
        norm = plt.Normalize(vmin=-vmax, vmax=vmax)

        # Graficar los círculos
        im = ax.scatter(x_coords, y_coords,
                        s=marker_size,
                        c=component_plane_values,
                        cmap=cmap,
                        marker='o',
                        edgecolor='black',
                        linewidths=0.5,
                        norm=norm)

        # Configurar Subplot
        feature_name = feature_names[i]
        ax.set_title(f'Component: {feature_name}', fontsize=12)

        # Ajustar límites y aspecto del gráfico
        margin = circle_radius * 1.05
        x_min = x_coords.min() - margin
        x_max = x_coords.max() + margin
        y_min = y_coords.min() - margin
        y_max = y_coords.max() + margin

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, aspect=10, shrink=0.8,
                     label=f'Standardized Weight ({feature_name})')

    for j in range(num_features, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle('All Component Planes - Compact Topology', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # Guardar gráfico
    results_dir = './results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    filename = f'{results_dir}/all_component_planes_{output_name}.png'
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def plot_bmu_table(mapping_df, map_rows, map_cols, output_name):
    """
    Genera y guarda una imagen PNG de la tabla de mapeo de países a BMU.

    :param mapping_df: DataFrame con las columnas 'Country', 'BMU_Row', 'BMU_Col'.
    :param map_rows: Número de filas del mapa.
    :param map_cols: Número de columnas del mapa.
    """
    df_sorted = mapping_df.sort_values(by=['BMU_Row', 'BMU_Col']).reset_index(drop=True)

    data_to_plot = df_sorted[['Country', 'BMU_Row', 'BMU_Col']].astype(str).values

    fig, ax = plt.subplots(figsize=(8, 1 + len(df_sorted) * 0.3))  # Tamaño ajustado
    ax.axis('off')  # Ocultar los ejes
    ax.axis('tight')  # Ajustar a los datos

    table = ax.table(
        cellText=data_to_plot,
        colLabels=['País', 'BMU Fila', 'BMU Columna'],
        colColours=["#f5f5f5"] * 3,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)  # Escalar el tamaño de las celdas

    ax.set_title(f'Mapeo de Países a Neurona Ganadora (BMU) en mapa {map_rows}x{map_cols}', fontsize=12, pad=20)

    filename = f'./results/bmu_mapping_table_{output_name}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Graph saved to: {filename}")
    plt.close(fig)