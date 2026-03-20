"""Clustering hyperparameter tuner for trained particle-gnn models.

Loads a trained model, extracts embeddings and MLP1 interaction curves,
and sweeps UMAP + blind clustering hyperparameters (no k given) to find
a single method that works across configs.

Usage:
    python tune_clustering.py arbitrary boids           # tune on these two
    python tune_clustering.py arbitrary boids gravity    # also report gravity
    python tune_clustering.py arbitrary                  # single config
"""

import argparse
import glob
import os
import warnings

import matplotlib
matplotlib.use('Agg')

import numpy as np
import torch
import umap

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import fclusterdata
from scipy.optimize import linear_sum_assignment

from particle_gnn.config import ParticleGNNConfig
from particle_gnn.models.utils import choose_training_model
from particle_gnn.utils import to_numpy, set_device, add_pre_folder, sort_key
from particle_gnn.plot import get_embedding, _batched_mlp_eval
from particle_gnn.zarr_io import load_simulation_data


def hungarian_accuracy(true_labels, cluster_labels):
    """Compute accuracy with optimal label mapping via Hungarian algorithm."""
    n_true = len(np.unique(true_labels))
    n_pred = len(np.unique(cluster_labels))
    size = max(n_true, n_pred)

    confusion = np.zeros((size, size))
    for t, c in zip(true_labels, cluster_labels):
        confusion[int(t), int(c)] += 1

    row_ind, col_ind = linear_sum_assignment(-confusion)
    mapping = {c: r for r, c in zip(row_ind, col_ind)}
    mapped = np.array([mapping.get(int(c), -1) for c in cluster_labels])
    return accuracy_score(true_labels, mapped)


def extract_features(model, config, device):
    """Extract embedding and MLP1 curves from a trained model."""
    sim = config.simulation
    dataset_name = config.dataset
    dimension = sim.dimension
    n_particles = sim.n_particles
    n_particle_types = sim.n_particle_types
    config_model = config.graph_model.particle_model_name
    max_radius = sim.max_radius

    from cell_gnn.utils import graphs_data_path
    x_ts = load_simulation_data(graphs_data_path(dataset_name, 'x_list_0'), dimension)
    type_list = to_numpy(x_ts.frame(0).particle_type[:n_particles]).astype(int)

    embedding = get_embedding(model.a, 0)[:n_particles]

    if 'gravity_ode' in config_model:
        rr = torch.tensor(np.linspace(0, max_radius * 1.3, 1000)).to(device)
        mlp_max_radius = max_radius
    elif 'boids_ode' in config_model:
        max_radius_plot = 0.04
        rr = torch.tensor(np.linspace(-max_radius_plot, max_radius_plot, 1000)).to(device)
        mlp_max_radius = max_radius_plot
    else:
        rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
        mlp_max_radius = max_radius

    if len(model.a.shape) == 3:
        all_embeddings = model.a[0, :n_particles, :]
    else:
        all_embeddings = model.a[:n_particles, :]

    func_list = _batched_mlp_eval(
        model.lin_edge, all_embeddings, rr,
        config_model, mlp_max_radius, device
    )
    func_list_np = to_numpy(func_list)

    return embedding, func_list_np, type_list, n_particle_types


def prepare_feature_sets(embedding, func_list):
    """Normalize and combine features into three feature sets."""
    scaler_emb = StandardScaler()
    emb_scaled = scaler_emb.fit_transform(embedding)

    scaler_func = StandardScaler()
    func_scaled = scaler_func.fit_transform(func_list)

    concatenated = np.column_stack([emb_scaled, func_scaled])

    return {
        'embedding_only': emb_scaled,
        'func_only': func_scaled,
        'concatenated': concatenated,
    }


def sweep_blind_clustering(feature_sets, type_list, n_particle_types, seed=42):
    """Sweep UMAP + blind clustering (no k given).

    Methods:
      - KMeans auto-k: sweep k=2..20, pick best silhouette
      - DBSCAN: sweep eps
      - Hierarchical distance: sweep threshold
    """
    umap_n_neighbors_list = [15, 30, 50, 100]
    umap_min_dist_list = [0.05, 0.1, 0.3]
    dbscan_eps_list = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    hier_thresholds = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]

    results = []
    best_result = None
    best_accuracy = -1.0

    n_combos = len(feature_sets) * len(umap_n_neighbors_list) * len(umap_min_dist_list)
    combo_idx = 0

    for feat_name, features in feature_sets.items():
        for n_neighbors in umap_n_neighbors_list:
            effective_nn = min(n_neighbors, features.shape[0] - 1)

            for min_dist in umap_min_dist_list:
                combo_idx += 1
                print(f'  [{combo_idx}/{n_combos}] {feat_name} nn={effective_nn} md={min_dist}', end='', flush=True)

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    reducer = umap.UMAP(
                        n_components=2,
                        n_neighbors=effective_nn,
                        min_dist=min_dist,
                        random_state=seed,
                    )
                    projected = reducer.fit_transform(features)

                local_best = 0.0

                def record(method_name, labels, n_found):
                    nonlocal local_best, best_accuracy, best_result
                    acc = hungarian_accuracy(type_list, labels)
                    r = dict(feature_set=feat_name, n_neighbors=effective_nn, min_dist=min_dist,
                             cluster_method=method_name, n_clusters_found=n_found, accuracy=acc)
                    results.append(r)
                    local_best = max(local_best, acc)
                    if acc > best_accuracy:
                        best_accuracy = acc
                        best_result = r

                # --- KMeans auto-k: pick k with best silhouette from 2..20 ---
                best_sil = -1.0
                best_k = 2
                for k in range(2, min(21, projected.shape[0])):
                    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
                    labels_k = km.fit_predict(projected)
                    if len(np.unique(labels_k)) < 2:
                        continue
                    sil = silhouette_score(projected, labels_k)
                    if sil > best_sil:
                        best_sil = sil
                        best_k = k
                        best_km_labels = labels_k
                record(f'KMeans_auto(k={best_k})', best_km_labels, best_k)

                # --- DBSCAN eps sweep ---
                for eps in dbscan_eps_list:
                    db = DBSCAN(eps=eps, min_samples=5)
                    db_labels = db.fit_predict(projected)
                    # remap noise (-1) to its own cluster
                    if -1 in db_labels:
                        db_labels = db_labels.copy()
                        db_labels[db_labels == -1] = db_labels.max() + 1
                    n_found = len(np.unique(db_labels))
                    record(f'DBSCAN(eps={eps})', db_labels, n_found)

                # --- Hierarchical distance threshold sweep ---
                for thresh in hier_thresholds:
                    hier_labels = fclusterdata(projected, thresh, criterion='distance', method='single') - 1
                    n_found = len(np.unique(hier_labels))
                    record(f'Hier(t={thresh})', hier_labels, n_found)

                print(f'  best={local_best:.4f}')

    return results, best_result


def load_config_and_model(config_name, best_model='best', device_override=None):
    """Load a config and its trained model. Returns (config, model, type_list, features)."""
    from cell_gnn.utils import config_path, log_path
    config_file, pre_folder = add_pre_folder(config_name)
    config = ParticleGNNConfig.from_yaml(config_path(f'{config_file}.yaml'))
    config.dataset = pre_folder + config.dataset
    config.config_file = pre_folder + config_name

    if device_override:
        device = device_override
    else:
        device = set_device(config.training.device)

    model, bc_pos, bc_dpos = choose_training_model(config, device)

    log_dir = log_path(config.config_file)
    ynorm = torch.load(f'{log_dir}/ynorm.pt', map_location=device, weights_only=True)
    vnorm = torch.load(f'{log_dir}/vnorm.pt', map_location=device, weights_only=True)
    if vnorm == 0:
        vnorm = ynorm
    model.ynorm = ynorm
    model.vnorm = vnorm

    n_runs = config.training.n_runs
    if best_model == 'best':
        files = glob.glob(f'{log_dir}/models/*')
        files.sort(key=sort_key)
        filename = files[-1].split('/')[-1].split('graphs')[-1][1:-3]
        best_model = filename

    net = f'{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{best_model}.pt'
    print(f'  Loading: {net}')
    state_dict = torch.load(net, map_location=device, weights_only=True)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    embedding, func_list, type_list, n_particle_types = extract_features(model, config, device)
    print(f'  embedding: {embedding.shape}, func_list: {func_list.shape}, '
          f'n_particles: {len(type_list)}, n_types: {n_particle_types}')

    feature_sets = prepare_feature_sets(embedding, func_list)
    return config_name, type_list, n_particle_types, feature_sets


def print_summary_table(results, best_result, top_n=20):
    """Print top results sorted by accuracy descending."""
    sorted_results = sorted(results, key=lambda r: r['accuracy'], reverse=True)

    header = f"{'Feature Set':<18} {'nn':>4} {'md':>6} {'Cluster Method':<22} {'k':>3} {'Accuracy':>8}"
    sep = '-' * len(header)

    print(f'\n{sep}')
    print('TOP RESULTS (blind clustering, no k given)')
    print(sep)
    print(header)
    print(sep)

    for r in sorted_results[:top_n]:
        marker = ' ***' if r is best_result else ''
        print(f"{r['feature_set']:<18} {r['n_neighbors']:>4} {r['min_dist']:>6.2f} "
              f"{r['cluster_method']:<22} {r['n_clusters_found']:>3} {r['accuracy']:>8.4f}{marker}")

    if len(sorted_results) > top_n:
        print(f'  ... ({len(sorted_results) - top_n} more rows omitted)')

    print(sep)
    if best_result:
        print(f'\nBEST: accuracy={best_result["accuracy"]:.4f}  '
              f'{best_result["cluster_method"]}  k_found={best_result["n_clusters_found"]}  '
              f'feat={best_result["feature_set"]}  nn={best_result["n_neighbors"]}  md={best_result["min_dist"]}')


def find_common_best(all_config_results):
    """Find the method+params combination with best minimum accuracy across configs.

    A 'method signature' is (feature_set, n_neighbors, min_dist, cluster_method).
    For each signature present in ALL configs, compute min(accuracy across configs).
    Return the signature with the highest min-accuracy.
    """
    # Group results by method signature
    from collections import defaultdict
    sig_results = defaultdict(dict)  # sig -> {config_name: accuracy}

    for config_name, results in all_config_results.items():
        for r in results:
            sig = (r['feature_set'], r['n_neighbors'], r['min_dist'], r['cluster_method'])
            sig_results[sig][config_name] = r

    # Find signatures present in all configs
    config_names = set(all_config_results.keys())
    best_min_acc = -1.0
    best_sig = None

    for sig, per_config in sig_results.items():
        if set(per_config.keys()) != config_names:
            continue
        min_acc = min(r['accuracy'] for r in per_config.values())
        if min_acc > best_min_acc:
            best_min_acc = min_acc
            best_sig = sig
            best_per_config = per_config

    return best_sig, best_per_config, best_min_acc


def main():
    parser = argparse.ArgumentParser(
        description='Tune blind clustering across multiple particle-gnn configs'
    )
    parser.add_argument('config_names', nargs='+',
                        help='Config names (e.g., "arbitrary boids gravity")')
    parser.add_argument('--best_model', type=str, default='best')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    # Load all configs and extract features
    all_data = {}
    for config_name in args.config_names:
        print(f'\n=== {config_name} ===')
        name, type_list, n_types, feature_sets = load_config_and_model(
            config_name, args.best_model, args.device)
        all_data[config_name] = (type_list, n_types, feature_sets)

    # Run sweep per config
    all_results = {}
    for config_name, (type_list, n_types, feature_sets) in all_data.items():
        print(f'\n=== Sweep: {config_name} (n_types={n_types}) ===')
        results, best = sweep_blind_clustering(feature_sets, type_list, n_types, seed=args.seed)
        all_results[config_name] = results

        print(f'\n--- {config_name} ---')
        print_summary_table(results, best)

    # Find best common method across all configs
    if len(all_results) > 1:
        best_sig, best_per_config, best_min_acc = find_common_best(all_results)

        print('\n' + '=' * 72)
        print('BEST COMMON METHOD (highest min-accuracy across all configs)')
        print('=' * 72)
        if best_sig:
            feat, nn, md, method = best_sig
            print(f'  feature_set: {feat}')
            print(f'  n_neighbors: {nn}')
            print(f'  min_dist:    {md}')
            print(f'  method:      {method}')
            print(f'  min_accuracy: {best_min_acc:.4f}')
            print()
            for cname, r in best_per_config.items():
                print(f'  {cname:<12} accuracy={r["accuracy"]:.4f}  k_found={r["n_clusters_found"]}')
        else:
            print('  No common signature found across all configs.')
        print('=' * 72)


if __name__ == '__main__':
    main()
