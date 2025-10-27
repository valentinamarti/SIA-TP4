import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional

def _get_staggered_coords(rows, cols):
    """
    Calculates the (x, y) coordinates on the plane for a staggered grid (like a hex grid).
    """
    x_coords = []
    y_coords = []

    spacing_x = 1.0
    spacing_y = spacing_x * (np.sqrt(3) / 2)  # Vertical scaling factor

    for r in range(rows):
        for c in range(cols):
            x = c * spacing_x + (0.5 * spacing_x * (r % 2))
            y = r * spacing_y

            x_coords.append(x)
            y_coords.append(y)

    return np.array(x_coords), np.array(y_coords)


def plot_circle_map(map_values, rows, cols, title, label, cmap, plot_type: str, filename: Optional[str] = None):
    """
    Generates a compact map using circles for U-Matrix, Hit Map, or Component Plane and saves it.
    Uses an aggressive marker size to ensure circles appear contiguous.
    """
    x_coords, y_coords = _get_staggered_coords(rows, cols)

    fig_width = (cols + 0.5 * (rows % 2)) * 0.7
    fig_height = rows * 0.7 * (np.sqrt(3) / 2)
    plt.figure(figsize=(fig_width, fig_height))

    norm = plt.Normalize(map_values.min(), map_values.max())
    marker_size = 8000 / (max(rows, cols))

    plt.scatter(x_coords, y_coords,
                s=marker_size,
                c=map_values,
                cmap=cmap,
                marker='o',
                edgecolor='black',
                linewidths=0.5,
                norm=norm)

    plt.colorbar(label=label, ax=plt.gca())
    plt.title(title, fontsize=14)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')

    if filename:
        results_dir = './results'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        save_path = os.path.join(results_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Graph saved to: {save_path}")
    else:
        plt.show(block=False)


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