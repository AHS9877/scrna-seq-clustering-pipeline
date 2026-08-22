"""
pipeline.py
-----------
A from-scratch single-cell RNA-seq analysis pipeline covering the same
stages as a standard Scanpy/Seurat workflow:

    1. Quality control (filter low-quality cells / rarely-detected genes)
    2. Normalization (library-size / CPM + log1p transform)
    3. Highly variable gene (HVG) selection (dispersion-based, Seurat-style)
    4. Dimensionality reduction (PCA)
    5. k-nearest-neighbour graph construction
    6. Graph-based clustering (greedy modularity community detection)
    7. 2D embedding for visualization (UMAP)
    8. Marker gene / differential expression detection per cluster
       (Welch's t-test + log fold change, Benjamini-Hochberg FDR)
    9. Cluster <-> ground-truth annotation agreement (ARI, NMI)

Run with:  python src/pipeline.py
Outputs land in results/ (figures + tables).
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from statsmodels.stats.multitest import multipletests

try:
    import umap
    HAVE_UMAP = True
except ImportError:
    HAVE_UMAP = False

sns.set_theme(style="whitegrid", font_scale=0.9)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_data():
    counts = pd.read_csv(DATA_DIR / "counts_matrix.csv", index_col=0)
    meta = pd.read_csv(DATA_DIR / "cell_metadata.csv", index_col="barcode")
    return counts, meta


def qc_filter(counts: pd.DataFrame, min_genes: int = 100, min_cells: int = 3):
    genes_per_cell = (counts > 0).sum(axis=1)
    counts_per_cell = counts.sum(axis=1)
    cells_per_gene = (counts > 0).sum(axis=0)

    keep_cells = genes_per_cell >= min_genes
    keep_genes = cells_per_gene >= min_cells

    print(f"QC: keeping {keep_cells.sum()}/{len(keep_cells)} cells, "
          f"{keep_genes.sum()}/{len(keep_genes)} genes")
    return counts.loc[keep_cells, keep_genes], counts_per_cell[keep_cells], genes_per_cell[keep_cells]


def normalize(counts: pd.DataFrame, target_sum: float = 1e4) -> pd.DataFrame:
    lib_sizes = counts.sum(axis=1)
    norm = counts.div(lib_sizes, axis=0) * target_sum
    return np.log1p(norm)


def select_hvgs(norm: pd.DataFrame, n_top: int = 500) -> list:
    mean = norm.mean(axis=0)
    var = norm.var(axis=0)
    dispersion = var / (mean + 1e-12)
    # bin genes by mean expression, z-score dispersion within each bin (Seurat approach)
    bins = pd.qcut(mean.rank(method="first"), q=20, labels=False)
    z = pd.Series(index=mean.index, dtype=float)
    for b in np.unique(bins):
        idx = bins == b
        d = dispersion[idx]
        z[idx] = (d - d.mean()) / (d.std() + 1e-12)
    return z.sort_values(ascending=False).head(n_top).index.tolist()


def run_pca(norm_hvg: pd.DataFrame, n_components: int = 30):
    X = norm_hvg.values
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    pca = PCA(n_components=n_components, random_state=0)
    pcs = pca.fit_transform(X)
    print(f"PCA: {n_components} PCs explain "
          f"{pca.explained_variance_ratio_.sum() * 100:.1f}% of variance")
    return pcs, pca


def build_knn_graph(pcs: np.ndarray, k: int = 15) -> nx.Graph:
    nn = NearestNeighbors(n_neighbors=k + 1).fit(pcs)
    _, indices = nn.kneighbors(pcs)
    G = nx.Graph()
    G.add_nodes_from(range(pcs.shape[0]))
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:
            G.add_edge(i, j)
    return G


def cluster_graph(G: nx.Graph) -> np.ndarray:
    communities = nx.algorithms.community.greedy_modularity_communities(G)
    labels = np.zeros(G.number_of_nodes(), dtype=int)
    for cluster_id, members in enumerate(communities):
        for node in members:
            labels[node] = cluster_id
    print(f"Clustering: found {len(communities)} clusters")
    return labels


def embed_umap(pcs: np.ndarray) -> np.ndarray:
    if HAVE_UMAP:
        reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.3)
        return reducer.fit_transform(pcs)
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, random_state=42).fit_transform(pcs)


def find_markers(norm: pd.DataFrame, labels: np.ndarray, top_n: int = 10) -> pd.DataFrame:
    records = []
    unique_clusters = np.unique(labels)
    for cl in unique_clusters:
        in_cluster = labels == cl
        group = norm.values[in_cluster]
        rest = norm.values[~in_cluster]
        t_stat, p_val = stats.ttest_ind(group, rest, axis=0, equal_var=False)
        log_fc = group.mean(axis=0) - rest.mean(axis=0)
        _, p_adj, _, _ = multipletests(np.nan_to_num(p_val, nan=1.0), method="fdr_bh")

        df = pd.DataFrame({
            "gene": norm.columns,
            "log_fc": log_fc,
            "p_adj": p_adj,
        })
        df = df[(df["p_adj"] < 0.05) & (df["log_fc"] > 0)].sort_values("log_fc", ascending=False).head(top_n)
        df.insert(0, "cluster", cl)
        records.append(df)
    return pd.concat(records, ignore_index=True)


def plot_results(embedding, labels, meta, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    sc1 = axes[0].scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=8, alpha=0.8)
    axes[0].set_title("Graph-based clusters (inferred)")
    axes[0].set_xlabel("UMAP 1"); axes[0].set_ylabel("UMAP 2")
    plt.colorbar(sc1, ax=axes[0], label="cluster")

    true_labels = meta["true_cell_type"].astype("category").cat.codes
    sc2 = axes[1].scatter(embedding[:, 0], embedding[:, 1], c=true_labels, cmap="tab10", s=8, alpha=0.8)
    axes[1].set_title("Ground-truth cell types")
    axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=ct,
               markerfacecolor=plt.cm.tab10(i / max(len(meta["true_cell_type"].unique())-1,1)), markersize=8)
               for i, ct in enumerate(meta["true_cell_type"].astype("category").cat.categories)]
    axes[1].legend(handles=handles, fontsize=7, loc="best")

    plt.tight_layout()
    plt.savefig(out_dir / "umap_clusters_vs_truth.png", dpi=150)
    plt.close()
    print(f"Saved figure: {out_dir / 'umap_clusters_vs_truth.png'}")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("STEP 1/8: Loading data")
    counts, meta = load_data()

    print("\nSTEP 2/8: Quality control")
    counts_qc, lib_sizes, genes_detected = qc_filter(counts)
    meta = meta.loc[counts_qc.index]

    print("\nSTEP 3/8: Normalization (CPM + log1p)")
    norm = normalize(counts_qc)

    print("\nSTEP 4/8: Highly variable gene selection")
    hvgs = select_hvgs(norm, n_top=500)
    norm_hvg = norm[hvgs]
    print(f"Selected {len(hvgs)} highly variable genes")

    print("\nSTEP 5/8: PCA")
    pcs, pca_model = run_pca(norm_hvg, n_components=30)

    print("\nSTEP 6/8: kNN graph + Louvain-style clustering")
    G = build_knn_graph(pcs, k=15)
    labels = cluster_graph(G)

    print("\nSTEP 7/8: UMAP embedding")
    embedding = embed_umap(pcs)

    print("\nSTEP 8/8: Marker gene detection + evaluation")
    markers = find_markers(norm, labels, top_n=10)
    markers.to_csv(RESULTS_DIR / "marker_genes_per_cluster.csv", index=False)
    print(f"Saved marker table: {RESULTS_DIR / 'marker_genes_per_cluster.csv'}")

    true_labels_codes = meta["true_cell_type"].astype("category").cat.codes.values
    ari = adjusted_rand_score(true_labels_codes, labels)
    nmi = normalized_mutual_info_score(true_labels_codes, labels)
    print(f"\nCluster agreement with ground truth -> ARI: {ari:.3f}, NMI: {nmi:.3f}")

    plot_results(embedding, labels, meta, RESULTS_DIR)

    summary = pd.DataFrame({"barcode": meta.index, "cluster": labels, "true_cell_type": meta["true_cell_type"].values})
    summary.to_csv(RESULTS_DIR / "cluster_assignments.csv", index=False)

    with open(RESULTS_DIR / "run_summary.txt", "w") as f:
        f.write(f"Cells after QC: {counts_qc.shape[0]}\n")
        f.write(f"Genes after QC: {counts_qc.shape[1]}\n")
        f.write(f"HVGs used: {len(hvgs)}\n")
        f.write(f"PCs used: 30 (explained variance: {pca_model.explained_variance_ratio_.sum()*100:.1f}%)\n")
        f.write(f"Clusters found: {len(np.unique(labels))}\n")
        f.write(f"ARI vs ground truth: {ari:.3f}\n")
        f.write(f"NMI vs ground truth: {nmi:.3f}\n")

    print("=" * 60)
    print("Pipeline complete. See results/ for outputs.")


if __name__ == "__main__":
    main()
