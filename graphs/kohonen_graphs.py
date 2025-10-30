import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional

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

    fig_width = 7.0
    fig_height = 6.5
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


    margin = circle_radius * 1.05
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


def plot_u_matrix(u_matrix, rows, cols):
    """Generates the U-Matrix plot (compact circles) and saves it."""
    plot_circle_map(u_matrix, rows, cols,
                    title='Unified Distance Matrix (U-Matrix) - Compact Topology',  # Título simplificado
                    label='Average Euclidean Distance (Dissimilarity)',
                    cmap='plasma',
                    plot_type='u_matrix',
                    filename='u_matrix.png')


def plot_hit_map(count_map, rows, cols):
    """Generates the Count Map plot (compact circles) and saves it."""
    plot_circle_map(count_map, rows, cols,
                    title='Count Map (Hit Frequency) - Compact Topology',  # Título simplificado
                    label='Number of Associated Countries (Density)',
                    cmap='coolwarm',
                    plot_type='hit_map',
                    filename='hit_map.png')


def plot_component_plane(weights, feature_index, feature_name, rows, cols):
    """Generates the Component Plane plot (compact circles) for a single feature and saves it."""
    component_plane = weights[:, feature_index]

    safe_feature_name = feature_name.replace('.', '_').replace(' ', '_')
    filename = f'component_{safe_feature_name}.png'

    plot_circle_map(component_plane, rows, cols,
                    title=f'Component Map: Influence of "{feature_name}"',  # Título simplificado
                    label=f'Standardized Weight (Z-Score) of {feature_name}',
                    cmap='coolwarm',
                    plot_type='component',
                    filename=filename)

def plot_all_component_planes(weights, feature_names, map_rows, map_cols):
    """
    Generates and saves heatmaps for all component planes (one for each feature).

    :param weights: The final weight matrix. (N_neurons x INPUT_DIM o map_rows x map_cols x INPUT_DIM).
    :param feature_names: List of feature names.
    :param map_rows: Number of grid rows.
    :param map_cols: Number of grid columns.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # === INICIO DE LA CORRECCIÓN ===
    # Si los pesos son 2D, se reestructuran a 3D: (map_rows, map_cols, N_features)
    if weights.ndim == 2:
        try:
            INPUT_DIM = weights.shape[1]
            weights = weights.reshape(map_rows, map_cols, INPUT_DIM)
        except ValueError:
            print(
                "Error: No se pudo reestructurar la matriz de pesos 2D a la forma (map_rows, map_cols, INPUT_DIM).")
            return

    if weights.ndim != 3:
        print("Error: La matriz de pesos no tiene la dimensión esperada (3D) después de intentar reestructurarla.")
        return

    num_features = weights.shape[2]
    # === FIN DE LA CORRECCIÓN ===

    # Determinar el layout de subplots
    cols = 3
    rows = int(np.ceil(num_features / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for i in range(num_features):
        ax = axes[i]

        # Esta línea ahora funciona porque 'weights' es 3D
        component_plane = weights[:, :, i]

        # Usar imshow para el mapa de calor
        im = ax.imshow(component_plane, cmap='viridis', aspect='auto')
        ax.set_title(feature_names[i], fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

        # Agregar colorbar a cada subplot
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, aspect=10, shrink=0.8)

    # Ocultar ejes no utilizados
    for j in range(num_features, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    filename = f'./results/all_component_planes.png'
    plt.savefig(filename)
    print(f"Graph saved to: {filename}")
    plt.close(fig)


def plot_bmu_table(mapping_df, map_rows, map_cols):
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

    filename = f'./results/bmu_mapping_table.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Graph saved to: {filename}")
    plt.close(fig)