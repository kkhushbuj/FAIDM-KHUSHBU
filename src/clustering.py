from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from .utils import header, save_fig
from .config import RANDOM_STATE, SILHOUETTE_SAMPLE_N, PCA_SAMPLE_N, K_MIN, K_MAX

def _sample_indices(n_rows: int, sample_n: int, seed: int):
    sample_n = min(sample_n, n_rows)
    rng = np.random.RandomState(seed)
    return rng.choice(n_rows, size=sample_n, replace=False)

def minibatch_kmeans_fit_and_report(X: pd.DataFrame, figures_dir: Path, reports_dir: Path):
    header("CLUSTERING — MiniBatchKMeans (quick)")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    n = Xs.shape[0]
    sil_idx = _sample_indices(n, SILHOUETTE_SAMPLE_N, RANDOM_STATE)
    pca_idx = _sample_indices(n, PCA_SAMPLE_N, RANDOM_STATE + 1)

    ks = list(range(K_MIN, K_MAX + 1))
    inertias, sils = [], []

    
    for k in ks:
        mbk = MiniBatchKMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            batch_size=4096,
            n_init=3,
            max_iter=200
        )
        labels = mbk.fit_predict(Xs)
        inertias.append(mbk.inertia_)

       
        sil = silhouette_score(Xs[sil_idx], labels[sil_idx])
        sils.append(float(sil))

   
    plt.figure()
    plt.plot(ks, inertias, marker="o")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("MiniBatchKMeans Elbow Curve")
    save_fig(figures_dir / "kmeans_elbow.png")

   
    plt.figure()
    plt.plot(ks, sils, marker="o")
    plt.xlabel("k")
    plt.ylabel("Silhouette (sample-based)")
    plt.title("MiniBatchKMeans Silhouette vs k (sample)")
    save_fig(figures_dir / "kmeans_silhouette.png")

    best_k = ks[int(np.argmax(sils))]

   
    mbk_final = MiniBatchKMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        batch_size=4096,
        n_init=5,
        max_iter=300
    )
    labels = mbk_final.fit_predict(Xs)

  
    sil_best = float(silhouette_score(Xs[sil_idx], labels[sil_idx]))
    db = float(davies_bouldin_score(Xs, labels))
    ch = float(calinski_harabasz_score(Xs, labels))

    metrics_text = "\n".join([
        f"Selected k: {best_k}",
        f"Silhouette (sample-based): {sil_best:.4f} (higher better)",
        f"Davies-Bouldin (full): {db:.4f} (lower better)",
        f"Calinski-Harabasz (full): {ch:.2f} (higher better)",
        f"Silhouette sample size: {len(sil_idx)}",
        f"PCA plot sample size: {len(pca_idx)}",
        "Clustering algorithm: MiniBatchKMeans (used for speed on large dataset)."
    ])

    print(metrics_text)
    (reports_dir / "clustering_kmeans_metrics.txt").write_text(metrics_text, encoding="utf-8")
    (reports_dir / "kmeans_best_k.txt").write_text(f"best_k={best_k}\n", encoding="utf-8")

    
    X_with = X.copy()
    X_with["cluster"] = labels
    profile = X_with.groupby("cluster").mean(numeric_only=True).round(3)
    profile.to_csv(reports_dir / "kmeans_cluster_profile.csv")

    
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    Xp = pca.fit_transform(Xs[pca_idx])
    lbl_p = labels[pca_idx]

    plt.figure()
    plt.scatter(Xp[:, 0], Xp[:, 1], c=lbl_p, s=8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Clusters (PCA 2D, sample)")
    save_fig(figures_dir / "kmeans_pca_scatter.png")

    return labels, best_k, profile
