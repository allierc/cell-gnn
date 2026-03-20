"""Test clustering pipeline on a saved arbitrary model.

Usage:
    python test_clustering.py                           # use latest checkpoint
    python test_clustering.py best_model_with_0_graphs_0  # specific checkpoint
"""
import sys
import os
import numpy as np
import torch
import umap
import warnings
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment

from cell_gnn.config import CellGNNConfig
from cell_gnn.models.utils import choose_training_model, get_type_list
from cell_gnn.plot import get_embedding
from cell_gnn.sparsify import EmbeddingCluster, sparsify_cluster
from cell_gnn.utils import to_numpy, config_path, log_path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load config and model ---
config = CellGNNConfig.from_yaml(config_path('arbitrary/arbitrary.yaml'))
n_cells = config.simulation.n_cells
n_cell_types = config.simulation.n_cell_types
dataset_name = config.dataset
log_dir = log_path(dataset_name, dataset_name)

model, bc_pos, bc_dpos = choose_training_model(config, device)

# Find checkpoint
if len(sys.argv) > 1:
    ckpt_name = sys.argv[1]
else:
    # Find latest checkpoint
    model_dir = os.path.join(log_dir, 'models')
    ckpts = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    # Sort by numbers in filename: best_model_with_0_graphs_EPOCH_BATCH.pt or _EPOCH.pt
    import re
    def sort_key(f):
        nums = re.findall(r'\d+', f)
        return tuple(int(n) for n in nums) if nums else (0,)
    ckpts.sort(key=sort_key)
    ckpt_name = ckpts[-1].replace('.pt', '')

net = f"{log_dir}/models/{ckpt_name}.pt"
print(f'Loading: {net}')
state_dict = torch.load(net, map_location=device, weights_only=True)
model.load_state_dict(state_dict['model_state_dict'])
model.eval()

# --- Extract embedding ---
embedding = get_embedding(model.a, 0)
print(f'Embedding shape: {embedding.shape}')
print(f'Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]')
print(f'Embedding std: {embedding.std():.4f}')

# --- Build type_list ---
type_list = torch.zeros(int(n_cells / n_cell_types), device=device)
for n in range(1, n_cell_types):
    type_list = torch.cat((type_list, n * torch.ones(int(n_cells / n_cell_types), device=device)), 0)
type_list = type_list.long()
type_np = to_numpy(type_list).flatten().astype(int)

# --- Hungarian accuracy helper ---
def hungarian_accuracy(true_labels, cluster_labels):
    n_true = len(np.unique(true_labels))
    n_found = len(np.unique(cluster_labels))
    size = max(n_true, n_found)
    confusion = np.zeros((size, size))
    for t, c in zip(true_labels, cluster_labels):
        confusion[int(t), int(c)] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion)
    mapping = {col_ind[i]: row_ind[i] for i in range(len(col_ind))}
    mapped = np.array([mapping.get(int(l), -1) for l in cluster_labels])
    return accuracy_score(true_labels, mapped)

# ============================================================
# Sweep UMAP params × DBSCAN eps
# ============================================================
print('\n=== UMAP + DBSCAN sweep on embedding ===\n')

umap_params = [
    (15, 0.05),
    (15, 0.1),
    (30, 0.1),
    (30, 0.3),
    (50, 0.1),
    (50, 0.3),
    (100, 0.1),
    (100, 0.3),
]
eps_values = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

best_acc = 0
best_params = None

for nn, md in umap_params:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            reducer = umap.UMAP(n_neighbors=nn, min_dist=md, n_components=2,
                                random_state=config.training.seed)
            proj = reducer.fit_transform(embedding)
        except Exception as e:
            print(f'  UMAP(nn={nn}, md={md}) failed: {e}')
            continue

    for eps in eps_values:
        db = DBSCAN(eps=eps, min_samples=5)
        labels = db.fit_predict(proj)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)

        # Clean labels for accuracy
        labels_clean = labels.copy()
        if -1 in labels_clean:
            labels_clean[labels_clean == -1] = n_clusters

        acc = hungarian_accuracy(type_np, labels_clean)

        marker = ' ***' if acc > best_acc else ''
        if acc > best_acc:
            best_acc = acc
            best_params = (nn, md, eps, n_clusters)

        print(f'  UMAP(nn={nn:3d}, md={md:.2f}) + DBSCAN(eps={eps:.1f}): '
              f'acc={acc:.3f}  clusters={n_clusters:3d}  noise={n_noise}{marker}')

print(f'\nBest: UMAP(nn={best_params[0]}, md={best_params[1]}) + '
      f'DBSCAN(eps={best_params[2]}) → acc={best_acc:.3f}, '
      f'clusters={best_params[3]}')

# ============================================================
# Also test: raw embedding (no UMAP) + DBSCAN
# ============================================================
print('\n=== Raw embedding + DBSCAN (no UMAP) ===\n')
emb_norm = (embedding - embedding.min()) / (embedding.max() - embedding.min() + 1e-10)
for eps in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]:
    db = DBSCAN(eps=eps, min_samples=5)
    labels = db.fit_predict(emb_norm)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    labels_clean = labels.copy()
    if -1 in labels_clean:
        labels_clean[labels_clean == -1] = n_clusters
    acc = hungarian_accuracy(type_np, labels_clean)
    print(f'  DBSCAN(eps={eps:.2f}): acc={acc:.3f}  clusters={n_clusters}')

# ============================================================
# Also test: KMeans (knows n_types)
# ============================================================
print('\n=== KMeans on raw embedding (k=n_cell_types) ===\n')
from sklearn.cluster import KMeans
km = KMeans(n_clusters=n_cell_types, random_state=42, n_init=10)
labels_km = km.fit_predict(embedding)
acc_km = hungarian_accuracy(type_np, labels_km)
print(f'  KMeans(k={n_cell_types}): acc={acc_km:.3f}')

# Also KMeans on UMAP(100, 0.3)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    proj100 = umap.UMAP(n_neighbors=100, min_dist=0.3, n_components=2,
                         random_state=config.training.seed).fit_transform(embedding)
labels_km2 = km.fit_predict(proj100)
acc_km2 = hungarian_accuracy(type_np, labels_km2)
print(f'  KMeans(k={n_cell_types}) on UMAP(100,0.3): acc={acc_km2:.3f}')

# ============================================================
# Test the new pipeline (matches plot_training_summary_panels)
# ============================================================
print('\n=== New pipeline: UMAP(100,0.3) + DBSCAN(0.3) direct (no normalization) ===\n')
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    proj_new = umap.UMAP(n_neighbors=100, min_dist=0.3, n_components=2,
                          random_state=config.training.seed).fit_transform(embedding)
db_new = DBSCAN(eps=0.3, min_samples=5)
labels_new = db_new.fit_predict(proj_new)
n_clusters_new = len(set(labels_new)) - (1 if -1 in labels_new else 0)
if -1 in labels_new:
    labels_new[labels_new == -1] = n_clusters_new
    n_clusters_new += 1
acc_new = hungarian_accuracy(type_np, labels_new)
print(f'  acc={acc_new:.3f}  clusters={n_clusters_new}')
