"""
simulate_data.py
-----------------
Generates a synthetic single-cell RNA-seq count matrix that mimics the
statistical properties of real droplet-based (10x Genomics) data:
    - Negative-binomial distributed counts (overdispersion, like real UMI data)
    - Multiple ground-truth cell populations, each defined by a set of
      marker genes that are up-regulated relative to background
    - Library-size variation and dropout, both hallmarks of real scRNA-seq

This lets the pipeline in `pipeline.py` be fully reproducible and run
offline, while still exercising every step a real analysis would need
(QC, normalization, HVG selection, PCA, graph clustering, UMAP,
differential expression / marker detection).

Swap this module out for `scanpy.read_10x_mtx()` or `sc.read_h5ad()`
to run the exact same pipeline on real data.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

CELL_TYPES = {
    "CD4_T_cell":      {"n_cells": 350, "n_markers": 25},
    "CD8_T_cell":      {"n_cells": 280, "n_markers": 25},
    "B_cell":          {"n_cells": 220, "n_markers": 30},
    "Monocyte":        {"n_cells": 300, "n_markers": 35},
    "NK_cell":         {"n_cells": 150, "n_markers": 20},
}

N_GENES = 2000
N_HOUSEKEEPING = 200  # highly expressed in all cells, not markers


def simulate(n_genes: int = N_GENES, out_dir: str = "data") -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    gene_names = [f"GENE_{i:04d}" for i in range(n_genes)]
    housekeeping_idx = RNG.choice(n_genes, N_HOUSEKEEPING, replace=False)

    # Assign a disjoint block of marker genes to each cell type
    remaining = [i for i in range(n_genes) if i not in housekeeping_idx]
    RNG.shuffle(remaining)
    marker_blocks = {}
    cursor = 0
    for ct, cfg in CELL_TYPES.items():
        marker_blocks[ct] = remaining[cursor: cursor + cfg["n_markers"]]
        cursor += cfg["n_markers"]

    all_counts = []
    all_labels = []
    all_barcodes = []
    cell_counter = 0

    for ct, cfg in CELL_TYPES.items():
        n_cells = cfg["n_cells"]
        # baseline mean expression per gene (log-normal, like real data)
        base_mean = RNG.lognormal(mean=0.3, sigma=1.0, size=n_genes)
        base_mean[housekeeping_idx] *= 8  # housekeeping genes: high expression
        base_mean[marker_blocks[ct]] *= RNG.uniform(6, 15, size=len(marker_blocks[ct]))

        # per-cell library size factor (real cells vary widely in depth)
        lib_size_factor = RNG.lognormal(mean=0, sigma=0.4, size=n_cells)

        # negative binomial dispersion (overdispersion typical of UMI counts)
        dispersion = 0.15

        counts = np.zeros((n_cells, n_genes), dtype=np.int32)
        for c in range(n_cells):
            mu = base_mean * lib_size_factor[c]
            p = dispersion / (dispersion + mu)
            n = mu * p / (1 - p)
            n = np.clip(n, 1e-6, None)
            counts[c] = RNG.negative_binomial(n, p)

        # simulate dropout (zero-inflation) - common in scRNA-seq
        dropout_mask = RNG.random((n_cells, n_genes)) < 0.08
        counts[dropout_mask] = 0

        all_counts.append(counts)
        all_labels.extend([ct] * n_cells)
        all_barcodes.extend([f"CELL_{cell_counter + i:05d}" for i in range(n_cells)])
        cell_counter += n_cells

    counts_matrix = np.vstack(all_counts)

    df = pd.DataFrame(counts_matrix, index=all_barcodes, columns=gene_names)
    df.to_csv(out_path / "counts_matrix.csv")

    meta = pd.DataFrame({"barcode": all_barcodes, "true_cell_type": all_labels})
    meta.to_csv(out_path / "cell_metadata.csv", index=False)

    marker_records = []
    for ct, idxs in marker_blocks.items():
        for i in idxs:
            marker_records.append({"cell_type": ct, "gene": gene_names[i]})
    pd.DataFrame(marker_records).to_csv(out_path / "ground_truth_markers.csv", index=False)

    print(f"Simulated {counts_matrix.shape[0]} cells x {counts_matrix.shape[1]} genes")
    print(f"Cell type composition:\n{meta['true_cell_type'].value_counts()}")
    print(f"Saved to: {out_path.resolve()}")


if __name__ == "__main__":
    simulate()
