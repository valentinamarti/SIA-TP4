import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from europe.kohonen_net import KohonenNet
from graphs.kohonen_graphs import plot_u_matrix, plot_hit_map
from parser.parser import load_and_preprocess_data


def get_group_folder(exp_num):
    """Determina la carpeta de grupo según el número de experimento"""
    if exp_num in [1, 2]:
        return 'grupo1_inicializacion'
    elif exp_num in [3, 4, 5]:
        return 'grupo2_radio_adaptativo'
    elif exp_num in [6, 7, 8]:
        return 'grupo3_eta_adaptativo'
    elif exp_num in [9, 10, 11, 12]:
        return 'grupo4_tamano_mapa'
    else:
        return 'otros'


def fit_with_tracking(net, X, epochs, initial_eta, initial_radius, init_method='random', 
                      eta_adaptive=True, radius_adaptive=True, track_per_epoch=False):
    """
    Entrena la red Kohonen y opcionalmente rastrea QE y TE por época.
    """
    N_samples = X.shape[0]
    T = epochs * N_samples
    
    net._initialize_weights(X, init_method)
    
    qe_history = []
    te_history = []
    
    t = 0
    for epoch in range(epochs):
        for x_p in X[np.random.permutation(N_samples)]:
            radius, learning_rate = net._calculate_learning_params(
                t, T, initial_eta, initial_radius, eta_adaptive, radius_adaptive
            )
            
            bmu_idx = net._find_bmu(x_p)
            h_t = net._neighborhood_function(bmu_idx, radius)
            
            error = x_p - net.weights
            delta_W = learning_rate * h_t[:, np.newaxis] * error
            net.weights += delta_W
            t += 1
        
        # Calcular QE y TE al final de cada época si se requiere
        if track_per_epoch:
            qe = net.calculate_quantization_error(X)
            te = net.calculate_topographic_error(X)
            qe_history.append(qe)
            te_history.append(te)
    
    return net.weights, qe_history, te_history


def run_experiments_with_group_plots(filepath, exper1iments):
    """
    Ejecuta experimentos y genera gráficos específicos por grupo.
    """
    if not os.path.exists("./results"):
        os.makedirs("./results")
    
    print(f"\n🚀 Ejecutando {len(experiments)} experimentos de Kohonen con análisis por grupos...\n")
    
    # Cargar datos
    X_scaled, countries, feature_names = load_and_preprocess_data(filepath)
    INPUT_DIM = X_scaled.shape[1]
    print(f"Datos cargados. Países={len(countries)}, Features={INPUT_DIM}")
    
    all_results = []
    experiment_details = []
    group1_runs = []  # Para almacenar múltiples corridas del grupo 1
    
    # Ejecutar todos los experimentos
    for i, params in enumerate(experiments, start=1):
        exp_num = i
        exp_custom_name = params.get('exp_name', f'Exp_{i}')
        exp_name = f"exp_{i}_{params['map_rows']}x{params['map_cols']}_eta{params['initial_eta']}_{params['init_method']}"
        
        print(f"\n🔹 [{i}/{len(experiments)}] {exp_custom_name}")
        
        # Determinar si necesitamos tracking por época
        # Grupo 1 (1-2): Inicialización, Grupo 2 (3-5): Radio adaptativo, Grupo 3 (6-8): Eta adaptativo
        track_epochs = (i in [1, 2, 3, 4, 5, 6, 7, 8])
        
        # Determinar si es grupo 1 (necesita múltiples corridas)
        is_group1 = (i in [1, 2])
        num_runs = 20 if is_group1 else 1
        
        for run_num in range(1, num_runs + 1):
            if is_group1 and num_runs > 1:
                print(f"  🔄 Corrida {run_num}/{num_runs}...", end=' ')
            
            start_time = time.time()
            
            # Inicializar y entrenar red
            net = KohonenNet(params['map_rows'], params['map_cols'], INPUT_DIM)
            if track_epochs:
                final_weights, qe_history, te_history = fit_with_tracking(
                    net, X_scaled, params['epochs'], params['initial_eta'], 
                    params['initial_radius'], params.get('init_method', 'sample'),
                    params.get('eta_adaptive', True), params.get('radius_adaptive', True),
                    track_per_epoch=True
                )
            else:
                final_weights = net.fit(
                    X=X_scaled,
                    epochs=params['epochs'],
                    initial_eta=params['initial_eta'],
                    initial_radius=params['initial_radius'],
                    init_method=params.get('init_method', 'sample'),
                    eta_adaptive=params.get('eta_adaptive', True),
                    radius_adaptive=params.get('radius_adaptive', True)
                )
                qe_history = []
                te_history = []
            
            # Calcular métricas finales
            count_map = net.calculate_hit_map(X_scaled)
            u_matrix = net.calculate_u_matrix()
            qe = net.calculate_quantization_error(X_scaled)
            te = net.calculate_topographic_error(X_scaled)
            
            duration = round(time.time() - start_time, 2)
            
            # Guardar resultados de esta corrida
            run_result = {
                'exp_num': exp_num,
                'exp_name': exp_name,
                'exp_custom_name': exp_custom_name,
                'run_num': run_num if is_group1 else None,
                'qe': qe,
                'te': te,
                'duration': duration
            }
            
            if is_group1:
                group1_runs.append(run_result)
                if run_num % 5 == 0 or run_num == num_runs:
                    print(f"QE={qe:.4f}, TE={te:.4f}")
            else:
                print(f"  ✓ QE={qe:.4f}, TE={te:.4f}, tiempo={duration}s")
            
            # Agregar a resultados generales (todas las corridas)
            all_results.append({
                'Experiment': exp_name,
                'Custom_Name': exp_custom_name,
                'Run': run_num if is_group1 else 1,
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
            
            # Guardar gráficos individuales solo en la primera corrida (o única corrida)
            if run_num == 1 or not is_group1:
                # Crear carpeta de grupo si no existe
                group_folder = get_group_folder(exp_num)
                group_path = f"./results/{group_folder}"
                if not os.path.exists(group_path):
                    os.makedirs(group_path)
                
                # Guardar gráficos individuales (U-Matrix y Hit Map) en carpeta de grupo
                plot_u_matrix(u_matrix, params['map_rows'], params['map_cols'], exp_name)
                plot_hit_map(count_map, params['map_rows'], params['map_cols'], exp_name)
                
                # Mover archivos generados a la carpeta del grupo
                u_matrix_file = f"./results/u_matrix_{exp_name}.png"
                hit_map_file = f"./results/hit_map_{exp_name}.png"
                if os.path.exists(u_matrix_file):
                    os.rename(u_matrix_file, f"{group_path}/u_matrix_{exp_name}.png")
                if os.path.exists(hit_map_file):
                    os.rename(hit_map_file, f"{group_path}/hit_map_{exp_name}.png")
                
                # Guardar detalles del experimento (solo primera corrida)
                experiment_details.append({
                    'exp_num': exp_num,
                    'exp_name': exp_name,
                    'exp_custom_name': exp_custom_name,
                    'params': params,
                    'net': net,
                    'u_matrix': u_matrix,
                    'hit_map': count_map,
                    'qe': qe,
                    'te': te,
                    'qe_history': qe_history,
                    'te_history': te_history,
                    'duration': duration
                })
    
    # Guardar CSV con resultados
    results_df = pd.DataFrame(all_results)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    csv_path = f"./results/kohonen_experiments_results_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ Resultados guardados en {csv_path}")
    
    # ============================================================
    # GENERAR GRÁFICOS POR GRUPO
    # ============================================================
    sns.set(style="whitegrid", context="talk")
    
    # --- GRUPO 1: Inicialización (Experimentos 1-2) ---
    init_exps = [exp for exp in experiment_details if exp['exp_num'] in [1, 2]]
    if len(init_exps) == 2:
        plot_init_comparison(init_exps, results_df, group1_runs)
    
    # --- GRUPO 2: Adaptación del Radio (Experimentos 3-5) ---
    radius_exps = [exp for exp in experiment_details if exp['exp_num'] in [3, 4, 5]]
    if len(radius_exps) >= 2:
        plot_radius_comparison(radius_exps, results_df)
    
    # --- GRUPO 3: Adaptación de Eta (Experimentos 6-8) ---
    eta_exps = [exp for exp in experiment_details if exp['exp_num'] in [6, 7, 8]]
    if len(eta_exps) >= 2:
        plot_eta_comparison(eta_exps, results_df)
    
    # --- GRUPO 4: Tamaño del Mapa (Experimentos 7-10) ---
    size_exps = [exp for exp in experiment_details if exp['exp_num'] in [9, 10, 11, 12]]
    if len(size_exps) >= 2:
        plot_size_comparison(size_exps, results_df)
    
    print("\n✅ Todos los gráficos por grupo generados en ./results/")
    
    return results_df


def plot_init_comparison(experiments, results_df, group1_runs=None):
    """Gráficos para comparación de inicialización (Random vs Sample) con múltiples corridas"""
    print("\n📊 Generando gráficos de comparación de inicialización...")
    
    # Crear carpeta del grupo
    group_folder = './results/grupo1_inicializacion'
    if not os.path.exists(group_folder):
        os.makedirs(group_folder)
    
    # Filtrar resultados (experimentos 1-2)
    init_df = results_df[
        results_df['Experiment'].str.extract(r'exp_(\d+)_')[0].astype(int).isin([1, 2])
    ]
    
    # Si hay múltiples corridas, crear DataFrame con todas las corridas
    if group1_runs and len(group1_runs) > 0:
        runs_df = pd.DataFrame(group1_runs)
        
        # === GRÁFICOS CON DISTRIBUCIÓN DE 20 CORRIDAS ===
        
        # Boxplot QE
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=runs_df, x='exp_custom_name', y='qe', hue='exp_custom_name', 
                   palette='Blues_r', legend=False)
        sns.stripplot(data=runs_df, x='exp_custom_name', y='qe', color='black', alpha=0.3, size=3)
        plt.title('Distribución de QE - 20 Corridas\nInicialización: Random vs Sample')
        plt.xlabel('Método de Inicialización')
        plt.ylabel('Quantization Error (QE)')
        plt.tight_layout()
        plt.savefig(f'{group_folder}/QE_boxplot_20runs.png', bbox_inches='tight', dpi=150)
        plt.close()
        
        # Boxplot TE
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=runs_df, x='exp_custom_name', y='te', hue='exp_custom_name', 
                   palette='Reds_r', legend=False)
        sns.stripplot(data=runs_df, x='exp_custom_name', y='te', color='black', alpha=0.3, size=3)
        plt.title('Distribución de TE - 20 Corridas\nInicialización: Random vs Sample')
        plt.xlabel('Método de Inicialización')
        plt.ylabel('Topographic Error (TE)')
        plt.tight_layout()
        plt.savefig(f'{group_folder}/TE_boxplot_20runs.png', bbox_inches='tight', dpi=150)
        plt.close()
        
        # Violin plot combinado
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.violinplot(data=runs_df, x='exp_custom_name', y='qe', hue='exp_custom_name', 
                      palette='Blues_r', legend=False, ax=axes[0])
        axes[0].set_title('Distribución de QE - 20 Corridas')
        axes[0].set_xlabel('Método de Inicialización')
        axes[0].set_ylabel('Quantization Error (QE)')
        
        sns.violinplot(data=runs_df, x='exp_custom_name', y='te', hue='exp_custom_name', 
                      palette='Reds_r', legend=False, ax=axes[1])
        axes[1].set_title('Distribución de TE - 20 Corridas')
        axes[1].set_xlabel('Método de Inicialización')
        axes[1].set_ylabel('Topographic Error (TE)')
        
        plt.tight_layout()
        plt.savefig(f'{group_folder}/QE_TE_violin_20runs.png', bbox_inches='tight', dpi=150)
        plt.close()
        
        # Estadísticas resumidas
        stats_df = runs_df.groupby('exp_custom_name').agg({
            'qe': ['mean', 'std', 'min', 'max'],
            'te': ['mean', 'std', 'min', 'max']
        }).round(4)
        stats_df.to_csv(f'{group_folder}/estadisticas_20runs.csv')
        print(f"  📊 Estadísticas de 20 corridas guardadas en estadisticas_20runs.csv")
    
    # QE Barplot (promedio si hay múltiples corridas)
    plt.figure(figsize=(8, 6))
    if group1_runs and len(group1_runs) > 0:
        # Usar promedio de las corridas
        avg_df = runs_df.groupby('exp_custom_name')[['qe', 'te']].mean().reset_index()
        avg_df.columns = ['Custom_Name', 'QE', 'TE']
        std_df = runs_df.groupby('exp_custom_name')[['qe', 'te']].std().reset_index()
        std_df.columns = ['Custom_Name', 'QE_std', 'TE_std']
        
        # Combinar para tener todo en un solo DataFrame
        plot_df = avg_df.merge(std_df, on='Custom_Name')
        
        # Crear barplot manualmente con errorbars
        ax = plt.gca()
        x_pos = range(len(plot_df))
        bars = ax.bar(x_pos, plot_df['QE'], yerr=plot_df['QE_std'], 
                     color=sns.color_palette('Blues_r', len(plot_df)), 
                     capsize=5, alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(plot_df['Custom_Name'])
        plt.title('QE Promedio - 20 Corridas (con desviación estándar)\n↓ menor es mejor')
    else:
        sns.barplot(data=init_df, x='Custom_Name', y='QE', hue='Custom_Name', 
                   palette='Blues_r', legend=False)
        plt.title('Quantization Error (QE) - Inicialización: Random vs Sample\n↓ menor es mejor')
    plt.xlabel('Método de Inicialización')
    plt.ylabel('Quantization Error (QE)')
    plt.tight_layout()
    plt.savefig(f'{group_folder}/QE_comparison.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    # TE Barplot (promedio si hay múltiples corridas)
    plt.figure(figsize=(8, 6))
    if group1_runs and len(group1_runs) > 0:
        # Crear barplot manualmente con errorbars
        ax = plt.gca()
        x_pos = range(len(plot_df))
        bars = ax.bar(x_pos, plot_df['TE'], yerr=plot_df['TE_std'], 
                     color=sns.color_palette('Reds_r', len(plot_df)), 
                     capsize=5, alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(plot_df['Custom_Name'])
        plt.title('TE Promedio - 20 Corridas (con desviación estándar)\n↓ menor es mejor')
    else:
        sns.barplot(data=init_df, x='Custom_Name', y='TE', hue='Custom_Name', 
                   palette='Reds_r', legend=False)
        plt.title('Topographic Error (TE) - Inicialización: Random vs Sample\n↓ menor es mejor')
    plt.xlabel('Método de Inicialización')
    plt.ylabel('Topographic Error (TE)')
    plt.tight_layout()
    plt.savefig(f'{group_folder}/TE_comparison.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    # QE y TE por época (línea) - comparación temporal
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for exp in experiments:
        if exp['exp_num'] in [1, 2] and exp['qe_history'] and len(exp['qe_history']) > 0:
            epochs_range = range(1, len(exp['qe_history']) + 1)
            axes[0].plot(epochs_range, exp['qe_history'], 
                        label=exp['exp_custom_name'], linewidth=2, marker='o', markersize=3)
    
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Quantization Error (QE)')
    axes[0].set_title('QE por Época - Inicialización: Random vs Sample')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    for exp in experiments:
        if exp['exp_num'] in [1, 2] and exp['te_history'] and len(exp['te_history']) > 0:
            epochs_range = range(1, len(exp['te_history']) + 1)
            axes[1].plot(epochs_range, exp['te_history'], 
                        label=exp['exp_custom_name'], linewidth=2, marker='s', markersize=3)
    
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Topographic Error (TE)')
    axes[1].set_title('TE por Época - Inicialización: Random vs Sample')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{group_folder}/errors_per_epoch.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"  ✓ Grupo 1: Gráficos guardados en {group_folder}/")


def plot_radius_comparison(experiments, results_df):
    """Gráficos para comparación de adaptación de radio - análisis topológico y por época"""
    print("\n📊 Generando gráficos de comparación de adaptación de radio...")
    
    # Crear carpeta del grupo
    group_folder = './results/grupo2_radio_adaptativo'
    if not os.path.exists(group_folder):
        os.makedirs(group_folder)
    
    # Filtrar resultados (experimentos 3-5)
    radius_df = results_df[
        results_df['Experiment'].str.extract(r'exp_(\d+)_')[0].astype(int).isin([3, 4, 5])
    ]
    
    radius_exps = [exp for exp in experiments if exp['exp_num'] in [3, 4, 5]]
    
    # QE Barplot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=radius_df, x='Custom_Name', y='QE', palette='Blues_r')
    plt.title('Quantization Error (QE) - Adaptación de Radio\n↓ menor es mejor')
    plt.xlabel('Configuración de Radio')
    plt.ylabel('Quantization Error (QE)')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(f'{group_folder}/QE_comparison.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    # TE Barplot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=radius_df, x='Custom_Name', y='TE', palette='Reds_r')
    plt.title('Topographic Error (TE) - Adaptación de Radio\n↓ menor es mejor')
    plt.xlabel('Configuración de Radio')
    plt.ylabel('Topographic Error (TE)')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(f'{group_folder}/TE_comparison.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    # === ANÁLISIS TOPOLÓGICO: U-Matrix comparativo ===
    n_exps = len(radius_exps)
    if n_exps >= 2:
        # Ajustar layout según número de experimentos
        if n_exps == 2:
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        else:  # 3 o más experimentos
            fig, axes = plt.subplots(1, n_exps, figsize=(6*n_exps, 7))
        
        if n_exps == 1:
            axes = [axes]
        
        for idx, exp in enumerate(radius_exps):
            u_matrix = exp['u_matrix']
            
            im = axes[idx].imshow(u_matrix, cmap='viridis', aspect='auto')
            axes[idx].set_title(f'U-Matrix - {exp["exp_custom_name"]}\n(Adaptativo debería ser más suave)')
            axes[idx].set_xlabel('Columna')
            axes[idx].set_ylabel('Fila')
            plt.colorbar(im, ax=axes[idx], label='Distancia promedio')
        
        plt.tight_layout()
        plt.savefig(f'{group_folder}/umatrix_comparison.png', bbox_inches='tight', dpi=150)
        plt.close()
    
    # === ANÁLISIS POR ÉPOCA: QE y TE ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for exp in radius_exps:
        if exp['qe_history'] and len(exp['qe_history']) > 0:
            epochs_range = range(1, len(exp['qe_history']) + 1)
            axes[0].plot(epochs_range, exp['qe_history'], 
                        label=exp['exp_custom_name'], linewidth=2, marker='o', markersize=3)
    
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Quantization Error (QE)')
    axes[0].set_title('QE por Época - Adaptación de Radio\nConvergencia de error')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    for exp in radius_exps:
        if exp['te_history'] and len(exp['te_history']) > 0:
            epochs_range = range(1, len(exp['te_history']) + 1)
            axes[1].plot(epochs_range, exp['te_history'], 
                        label=exp['exp_custom_name'], linewidth=2, marker='s', markersize=3)
    
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Topographic Error (TE)')
    axes[1].set_title('TE por Época - Adaptación de Radio\nPreservación topológica')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{group_folder}/errors_per_epoch.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"  ✓ Grupo 2: Gráficos guardados en {group_folder}/")


def plot_eta_comparison(experiments, results_df):
    """Gráficos para comparación de adaptación de eta"""
    print("\n📊 Generando gráficos de comparación de adaptación de eta...")
    
    # Crear carpeta del grupo
    group_folder = './results/grupo3_eta_adaptativo'
    if not os.path.exists(group_folder):
        os.makedirs(group_folder)
    
    # Filtrar resultados (experimentos 6-8)
    eta_df = results_df[
        results_df['Experiment'].str.extract(r'exp_(\d+)_')[0].astype(int).isin([6, 7, 8])
    ]
    
    # QE por época (línea) - si tenemos historial
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for exp in experiments:
        if exp['qe_history'] and len(exp['qe_history']) > 0:
            epochs_range = range(1, len(exp['qe_history']) + 1)
            axes[0].plot(epochs_range, exp['qe_history'], 
                        label=exp['exp_custom_name'], linewidth=2, marker='o', markersize=3)
    
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Quantization Error (QE)')
    axes[0].set_title('QE por Época - Adaptación de η\nAdaptativo debe converger suavemente')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # TE por época (línea)
    for exp in experiments:
        if exp['te_history'] and len(exp['te_history']) > 0:
            epochs_range = range(1, len(exp['te_history']) + 1)
            axes[1].plot(epochs_range, exp['te_history'], 
                        label=exp['exp_custom_name'], linewidth=2, marker='s', markersize=3)
    
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Topographic Error (TE)')
    axes[1].set_title('TE por Época - Adaptación de η')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{group_folder}/errors_per_epoch.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    # Final QE/TE bars
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.barplot(data=eta_df, x='Custom_Name', y='QE', palette='Blues_r', ax=axes[0])
    axes[0].set_title('QE Final - Adaptación de η\n↓ menor es mejor')
    axes[0].set_xlabel('Configuración de η')
    axes[0].set_ylabel('Quantization Error (QE)')
    axes[0].tick_params(axis='x', rotation=15)
    
    sns.barplot(data=eta_df, x='Custom_Name', y='TE', palette='Reds_r', ax=axes[1])
    axes[1].set_title('TE Final - Adaptación de η\n↓ menor es mejor')
    axes[1].set_xlabel('Configuración de η')
    axes[1].set_ylabel('Topographic Error (TE)')
    axes[1].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig(f'{group_folder}/final_QE_TE.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"  ✓ Grupo 3: Gráficos guardados en {group_folder}/")


def plot_size_comparison(experiments, results_df):
    """Gráficos para comparación de tamaños de mapa - U-Matrix y Hit Map comparativos"""
    print("\n📊 Generando gráficos de comparación de tamaños de mapa...")
    
    # Crear carpeta del grupo
    group_folder = './results/grupo4_tamano_mapa'
    if not os.path.exists(group_folder):
        os.makedirs(group_folder)
    
    # Filtrar resultados (experimentos 7, 8, 9, 10)
    size_df = results_df[
        results_df['Experiment'].str.extract(r'exp_(\d+)_')[0].astype(int).isin([7, 8, 9, 10])
    ].copy()
    size_df = size_df.sort_values('MapSize')
    
    size_exps = [exp for exp in experiments if exp['exp_num'] in [7, 8, 9, 10]]
    size_exps = sorted(size_exps, key=lambda x: x['params']['map_rows'] * x['params']['map_cols'])
    
    # QE vs Tamaño
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=size_df, x='MapSize', y='QE', marker='o', linewidth=2, markersize=8)
    plt.title('Quantization Error (QE) vs Tamaño del Mapa\nQE ↓ con más neuronas (mejor ajuste)')
    plt.xlabel('Tamaño del Mapa (neurona)')
    plt.ylabel('Quantization Error (QE)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{group_folder}/QE_vs_size.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    # TE vs Tamaño
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=size_df, x='MapSize', y='TE', marker='s', linewidth=2, markersize=8, color='red')
    plt.title('Topographic Error (TE) vs Tamaño del Mapa\nTE puede subir si hay demasiadas neuronas')
    plt.xlabel('Tamaño del Mapa (neurona)')
    plt.ylabel('Topographic Error (TE)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{group_folder}/TE_vs_size.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    # QE y TE en un solo gráfico
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(size_df['MapSize'], size_df['QE'], 'o-', linewidth=2, markersize=8, label='QE', color='blue')
    ax1.set_xlabel('Tamaño del Mapa (neurona)')
    ax1.set_ylabel('Quantization Error (QE)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(size_df['MapSize'], size_df['TE'], 's-', linewidth=2, markersize=8, label='TE', color='red')
    ax2.set_ylabel('Topographic Error (TE)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title('QE y TE vs Tamaño del Mapa')
    plt.tight_layout()
    plt.savefig(f'{group_folder}/QE_TE_vs_size.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    # === U-MATRIX COMPARATIVO ===
    n_sizes = len(size_exps)
    if n_sizes > 0:
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()
        
        for idx, exp in enumerate(size_exps):
            if idx < 4:  # Máximo 4 tamaños
                u_matrix = exp['u_matrix']
                
                im = axes[idx].imshow(u_matrix, cmap='viridis', aspect='auto')
                axes[idx].set_title(f'U-Matrix - {exp["exp_custom_name"]}\nDetalle creciente y segmentación')
                axes[idx].set_xlabel('Columna')
                axes[idx].set_ylabel('Fila')
                plt.colorbar(im, ax=axes[idx], label='Distancia promedio')
        
        # Ocultar subplots no usados
        for idx in range(n_sizes, 4):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{group_folder}/umatrix_comparison.png', bbox_inches='tight', dpi=150)
        plt.close()
    
    # === HIT MAP COMPARATIVO ===
    if n_sizes > 0:
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()
        
        for idx, exp in enumerate(size_exps):
            if idx < 4:  # Máximo 4 tamaños
                hit_map = exp['hit_map']
                
                im = axes[idx].imshow(hit_map, cmap='YlOrRd', aspect='auto')
                axes[idx].set_title(f'Hit Map - {exp["exp_custom_name"]}\nCobertura (evitar neuronas muertas)')
                axes[idx].set_xlabel('Columna')
                axes[idx].set_ylabel('Fila')
                
                # Agregar valores en cada celda
                rows, cols = hit_map.shape
                for i in range(rows):
                    for j in range(cols):
                        text = axes[idx].text(j, i, int(hit_map[i, j]),
                                             ha="center", va="center", color="black", fontsize=8)
                
                # Colorbar con solo enteros
                from matplotlib.ticker import MaxNLocator
                cbar = plt.colorbar(im, ax=axes[idx], label='Número de activaciones')
                cbar.locator = MaxNLocator(integer=True)
                cbar.update_ticks()
        
        # Ocultar subplots no usados
        for idx in range(n_sizes, 4):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{group_folder}/hitmap_comparison.png', bbox_inches='tight', dpi=150)
        plt.close()
    
    print(f"  ✓ Grupo 4: Gráficos guardados en {group_folder}/")


# Definición de experimentos para análisis sistemático de parámetros de Kohonen
experiments = [
    # --- Inicialización: random vs sample
    {'map_rows': 4, 'map_cols': 4, 'epochs': 50,
     'initial_eta': 0.5, 'initial_radius': 2.0,
     'eta_adaptive': True, 'radius_adaptive': True,
     'init_method': 'random', 'exp_name': 'Init - Random'},
    {'map_rows': 4, 'map_cols': 4, 'epochs': 50,
     'initial_eta': 0.5, 'initial_radius': 2.0,
     'eta_adaptive': True, 'radius_adaptive': True,
     'init_method': 'sample', 'exp_name': 'Init - Sample'},

    # --- Adaptación del radio: activado vs desactivado
    {'map_rows': 4, 'map_cols': 4, 'epochs': 100,
     'initial_eta': 0.5, 'initial_radius': 3,
     'eta_adaptive': True, 'radius_adaptive': True,
     'init_method': 'sample', 'exp_name': 'Radio Adaptativo ON'},
    {'map_rows': 4, 'map_cols': 4, 'epochs': 100,
     'initial_eta': 0.5, 'initial_radius': 3,
     'eta_adaptive': True, 'radius_adaptive': False,
     'init_method': 'sample', 'exp_name': 'Radio Adaptativo OFF'},
    {'map_rows': 4, 'map_cols': 4, 'epochs': 100,
     'initial_eta': 0.5, 'initial_radius': 2,
     'eta_adaptive': True, 'radius_adaptive': False,
     'init_method': 'sample', 'exp_name': 'Radio Adaptativo OFF'},

    # --- Adaptación de la tasa de aprendizaje (eta): activado vs desactivado
    {'map_rows': 4, 'map_cols': 4, 'epochs': 100,
     'initial_eta': 0.5, 'initial_radius': 4.0,
     'eta_adaptive': True, 'radius_adaptive': True,
     'init_method': 'sample', 'exp_name': 'Eta Adaptativo ON'},
    {'map_rows': 4, 'map_cols': 4, 'epochs': 100,
     'initial_eta': 0.75, 'initial_radius': 4.0,
     'eta_adaptive': False, 'radius_adaptive': True,
     'init_method': 'sample', 'exp_name': 'Eta Adaptativo OFF'},
    {'map_rows': 4, 'map_cols': 4, 'epochs': 100,
     'initial_eta': 0.5, 'initial_radius': 4.0,
     'eta_adaptive': False, 'radius_adaptive': True,
     'init_method': 'sample', 'exp_name': 'Eta Adaptativo OFF'},

    # --- Tamaño del mapa: variaciones
    {'map_rows': 3, 'map_cols': 3, 'epochs': 100,
     'initial_eta': 0.5, 'initial_radius': 4.0,
     'eta_adaptive': True, 'radius_adaptive': True,
     'init_method': 'sample', 'exp_name': 'Mapa 3x3'},
    {'map_rows': 4, 'map_cols': 4, 'epochs': 100,
     'initial_eta': 0.5, 'initial_radius': 4.0,
     'eta_adaptive': True, 'radius_adaptive': True,
     'init_method': 'sample', 'exp_name': 'Mapa 4x4'},
    {'map_rows': 5, 'map_cols': 5, 'epochs': 100,
     'initial_eta': 0.5, 'initial_radius': 5.0,
     'eta_adaptive': True, 'radius_adaptive': True,
     'init_method': 'sample', 'exp_name': 'Mapa 5x5'},
    {'map_rows': 6, 'map_cols': 6, 'epochs': 100,
     'initial_eta': 0.5, 'initial_radius': 6.0,
     'eta_adaptive': True, 'radius_adaptive': True,
     'init_method': 'sample', 'exp_name': 'Mapa 6x6'},
]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python kohonen_experiments.py <ruta_al_archivo_csv>")
        sys.exit(1)

    DATA_FILEPATH = sys.argv[1]

    print(f"Total de experimentos a ejecutar: {len(experiments)}")

    # Ejecutar experimentos con análisis por grupos
    results_df = run_experiments_with_group_plots(DATA_FILEPATH, experiments)

    print("\n✅ Análisis completo de experimentos finalizado.")
    print(f"📊 Resultados y gráficos guardados en: ./results/")
