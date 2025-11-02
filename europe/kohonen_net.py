import numpy as np
import math
from sklearn.preprocessing import StandardScaler
import pandas as pd


class KohonenNet:
    """
    Implementation of the Kohonen Model (Self-Organizing Map - SOM).
    """
    def __init__(self, map_rows, map_cols, input_dim):
        """
        Initializes the Kohonen network.

        :param map_rows: Number of rows in the output grid.
        :param map_cols: Number of columns in the output grid.
        :param input_dim: Dimension of the input data (N).
        """
        self.rows = map_rows
        self.cols = map_cols
        self.input_dim = input_dim
        self.weights = None
        self.neuron_coords = np.array([
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
        ])

    def _initialize_weights(self, X, init_method='random'):
        """
        Initializes the network weights.

        :param X: Training data (used for sample-based initialization).
        :param init_method: 'random' (uniform [0, 1]) or 'sample' (data sampling).
        """
        num_samples = X.shape[0]
        num_neurons = self.rows * self.cols

        if init_method == 'random':
            self.weights = np.random.uniform(low=-3.0, high=3.0, size=(self.rows * self.cols, self.input_dim))
        elif init_method == 'sample':
            allow_replace = num_neurons > num_samples
            idx = np.random.choice(num_samples, size=num_neurons, replace=allow_replace)
            self.weights = X[idx, :]
        else:
            raise ValueError("Método de inicialización no válido.")


    def _find_bmu(self, x):
        """
        Finds the Best Matching Unit (BMU) using the minimum Euclidean Distance.
        """
        # Calculate euclidean distance between the input x y and all W weights
        distances = np.linalg.norm(self.weights - x, axis=1)

        # BMU is where distance is min
        bmu_index = np.argmin(distances)

        return bmu_index

    def _calculate_learning_params(self, t, T, initial_eta, initial_radius, eta_adaptive=True, radius_adaptive=True):
        """
        Calcula el radio de vecindario y la tasa de aprendizaje.

        Permite desactivar el decaimiento exponencial para fines de prueba.

        :param t: Current iteration (time step).
        :param T: Total number of update steps.
        :param initial_eta: Initial learning rate (eta_0).
        :param initial_radius: Initial neighborhood radius (R_0).
        :param eta_adaptive: Si es True, eta decae exponencialmente. Si es False, es constante = initial_eta.
        :param radius_adaptive: Si es True, R decae exponencialmente con piso de 1.0. Si es False, es constante = initial_radius.
        :return: current_radius, current_learning_rate
        """

        # 1. Inicialización con los valores por defecto (si no son adaptativos)
        radius = initial_radius
        learning_rate = initial_eta

        # 2. Solo calculamos el factor de decaimiento si AL MENOS uno es adaptativo
        if radius_adaptive or eta_adaptive:
            # Cálculo de sigma_0 para el decaimiento
            if initial_radius > 1.0:
                sigma_0 = T / np.log(initial_radius)
            else:
                sigma_0 = T

                # Cálculo del factor de decaimiento (común para R y eta)
            decaying_factor = np.exp(-t / sigma_0)

            # 3. Cálculo del radio (R) si es adaptativo
            if radius_adaptive:
                if initial_radius > 1.0:
                    calculated_radius = initial_radius * decaying_factor
                    # Restricción: el radio debe ser como mínimo 1.0
                    radius = max(1.0, calculated_radius)
                else:
                    # Si R0 ya era <= 1.0, se mantiene en 1.0
                    radius = 1.0

            # 4. Cálculo de la tasa de aprendizaje (eta) si es adaptativo
            if eta_adaptive:
                # La tasa de aprendizaje debe decaer a cero para garantizar la convergencia.
                learning_rate = initial_eta * decaying_factor

        # 5. Si ambos son False (no adaptativos), se devuelven initial_radius e initial_eta
        return radius, learning_rate


    def _neighborhood_function(self, bmu_index, current_radius):
        """
        Gaussian Neighborhood Function (H(t)).
        H(t) = exp(-d^2 / (2 * R(t)^2)), where d is the distance on the grid.
        """
        bmu_coords = self.neuron_coords[bmu_index]

        # Euclidean distance (d) between BMU y and all 2D neurons
        distance_to_bmu = np.linalg.norm(self.neuron_coords - bmu_coords, axis=1)

        if current_radius < 1e-6:
            h_t = np.zeros(self.rows * self.cols)
            h_t[bmu_index] = 1.0
            return h_t

        # H(t)): exp(-d^2 / (2 * R(t)^2))
        h_t = np.exp(-distance_to_bmu ** 2 / (2 * current_radius ** 2))
        return h_t


    def fit(self, X, epochs, initial_eta, initial_radius, init_method='random', eta_adaptive=True, radius_adaptive=True):
        """
        Trains the Kohonen Network with the dataset X.

        :param radius_adaptive:
        :param eta_adaptive:
        :param X: NumPy array of input data (standardized features).
        :param epochs: Number of passes over the entire dataset.
        :param initial_eta: Initial learning rate (eta_0).
        :param initial_radius: Initial neighborhood radius (R_0).
        :param init_method: Weight initialization method ('random' or 'sample').
        """
        N_samples = X.shape[0]
        T = epochs * N_samples

        self._initialize_weights(X, init_method)

        t = 0

        print(f"Iniciando la red de Kohonen ({self.rows}x{self.cols}). T_total={T} pasos.")

        for epoch in range(epochs):
            # Iterate over a shuffled view to avoid mutating the original X order
            for x_p in X[np.random.permutation(N_samples)]:
                # Calculate (R(t) y eta(t))
                radius, learning_rate = self._calculate_learning_params(
                    t, T, initial_eta, initial_radius, eta_adaptive, radius_adaptive
                )

                bmu_idx = self._find_bmu(x_p)
                h_t = self._neighborhood_function(bmu_idx, radius)


                error = x_p - self.weights
                # Delta_W = eta(t) * H(t) * (X_p - W)
                delta_W = learning_rate * h_t[:, np.newaxis] * error

                self.weights += delta_W
                t += 1

        print("Entrenamiento finalizado.")
        return self.weights


    def calculate_hit_map(self, X):
        """
        Calculates the Hit Map (Count Map).
        Returns a 2D array with the count of 'hits' per neuron.
        """
        if self.weights is None:
            raise RuntimeError("The network must be trained before calculating the hit map.")

        count_map = np.zeros((self.rows, self.cols), dtype=int)

        for x_p in X:
            bmu_index_1d = self._find_bmu(x_p)

            # Convert the linear index (1D) to grid coordinates (2D)
            bmu_row = bmu_index_1d // self.cols
            bmu_col = bmu_index_1d % self.cols

            # Increment the counter for the BMU's coordinates
            count_map[bmu_row, bmu_col] += 1

        return count_map


    def calculate_u_matrix(self):
        """
        Calculates the Unified Distance Matrix (U-Matrix).
        The U-Matrix shows the average Euclidean distance between a neuron's weight
        vector and its immediate neighbors' weight vectors.
        Returns the U-Matrix (2D array of average distances).
        """
        if self.weights is None:
            raise RuntimeError("The network must be trained before calculating the U-Matrix.")

        u_matrix = np.zeros((self.rows, self.cols))

        # Iterate over every neuron in the grid
        for r in range(self.rows):
            for c in range(self.cols):
                neuron_index_1d = r * self.cols + c
                current_weight = self.weights[neuron_index_1d]
                neighbor_distances = []

                # Check the 4 immediate neighbors (Up, Down, Left, Right)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc

                    # Check if the neighbor is within grid boundaries
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        neighbor_index_1d = nr * self.cols + nc
                        neighbor_weight = self.weights[neighbor_index_1d]

                        # Euclidean distance between the current weight and the neighbor's weight
                        distance = np.linalg.norm(current_weight - neighbor_weight)
                        neighbor_distances.append(distance)

                # The U-Matrix value is the average of distances to neighbors
                if neighbor_distances:
                    u_matrix[r, c] = np.mean(neighbor_distances)

        return u_matrix


    def map_data_to_bmus(self, X, countries):
        """
        Calculates the Country -> BMU (row, column) association.
        Returns a Pandas DataFrame with the association.
        """
        if self.weights is None:
            raise RuntimeError("The network must be trained before mapping the data.")

        mapping_data = {'Country': [], 'BMU_Row': [], 'BMU_Col': []}

        for i, x_p in enumerate(X):
            bmu_index_1d = self._find_bmu(x_p)

            bmu_row = bmu_index_1d // self.cols
            bmu_col = bmu_index_1d % self.cols

            mapping_data['Country'].append(countries[i])
            mapping_data['BMU_Row'].append(bmu_row)
            mapping_data['BMU_Col'].append(bmu_col)

        return pd.DataFrame(mapping_data)

    def calculate_quantization_error(self, X):
        """
        Calculates the Quantization Error (QE).
        QE = average Euclidean distance between each input and its BMU weight vector.
        """
        if self.weights is None:
            raise RuntimeError("Train the network before computing Quantization Error.")

        total_error = 0.0
        for x_p in X:
            bmu_idx = self._find_bmu(x_p)
            w_bmu = self.weights[bmu_idx]
            total_error += np.linalg.norm(x_p - w_bmu)

        qe = total_error / X.shape[0]
        return qe


    def calculate_topographic_error(self, X):
        """
        Calculates the Topographic Error (TE).
        TE = fraction of samples whose first and second BMUs are not adjacent on the grid.
        """
        if self.weights is None:
            raise RuntimeError("Train the network before computing Topographic Error.")

        num_samples = X.shape[0]
        error_count = 0

        for x_p in X:
            # Compute distances to all neurons
            distances = np.linalg.norm(self.weights - x_p, axis=1)
            bmu_indices = np.argsort(distances)[:2]  # Best and 2nd best
            bmu1 = self.neuron_coords[bmu_indices[0]]
            bmu2 = self.neuron_coords[bmu_indices[1]]

            # Manhattan distance between both BMUs
            grid_distance = np.abs(bmu1[0] - bmu2[0]) + np.abs(bmu1[1] - bmu2[1])

            if grid_distance > 1:
                error_count += 1

        te = error_count / num_samples
        return te
