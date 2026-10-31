"""Clustering and PCA analysis using the scikit-learn Wine dataset.

Algorithms:
    1. K-Means clustering
    2. Gaussian Mixture Model (GMM)
    3. Agglomerative (Hierarchical) clustering
    4. Principal Component Analysis (PCA)

The Wine class labels are NOT used to train the clustering models. They are
used only after clustering to calculate external validation scores.
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
N_CLUSTERS = 3  # The Wine dataset contains three reference classes.
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def load_and_prepare_data():
    """Load the built-in Wine dataset and standardize its numeric features."""
    wine = load_wine(as_frame=True)
    X_df = wine.data.copy()
    y_true = wine.target.to_numpy()

    # Basic data checks and preprocessing.
    missing_before = int(X_df.isna().sum().sum())
    if missing_before:
        X_df = X_df.fillna(X_df.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    print("\nDATASET INFORMATION")
    print("-" * 68)
    print(f"Dataset             : sklearn Wine dataset")
    print(f"Observations        : {X_df.shape[0]}")
    print(f"Numeric features    : {X_df.shape[1]}")
    print(f"Reference classes   : {len(np.unique(y_true))}")
    print(f"Missing values      : {missing_before}")
    print("Preprocessing       : StandardScaler (zero mean, unit variance)")
    print("Target labels       : Reserved for evaluation; not used for fitting")

    return wine, X_df, X_scaled, y_true


def fit_clustering_models(X_scaled):
    """Fit K-Means, GMM and agglomerative hierarchical clustering."""
    models = {
        "K-Means": KMeans(
            n_clusters=N_CLUSTERS,
            random_state=RANDOM_STATE,
            n_init=20,
        ),
        "GMM": GaussianMixture(
            n_components=N_CLUSTERS,
            covariance_type="full",
            random_state=RANDOM_STATE,
            n_init=5,
        ),
        "Hierarchical": AgglomerativeClustering(
            n_clusters=N_CLUSTERS,
            linkage="ward",
        ),
    }

    labels = {}
    for name, model in models.items():
        labels[name] = model.fit_predict(X_scaled)
        print(f"Fitted: {name}")

    return models, labels


def calculate_validation_metrics(X_scaled, y_true, labels, models):
    """Calculate internal and external clustering validation measures."""
    rows = []

    for name, cluster_labels in labels.items():
        # Internal metrics assess cluster geometry without reference labels.
        sil = silhouette_score(X_scaled, cluster_labels)
        db = davies_bouldin_score(X_scaled, cluster_labels)
        ch = calinski_harabasz_score(X_scaled, cluster_labels)

        # External metrics compare the clustering with known Wine classes.
        # They are calculated only after the unsupervised models are fitted.
        row = {
            "Algorithm": name,
            "Silhouette Score (higher is better)": sil,
            "Davies-Bouldin Index (lower is better)": db,
            "Calinski-Harabasz Score (higher is better)": ch,
            "Adjusted Rand Index (higher is better)": adjusted_rand_score(
                y_true, cluster_labels
            ),
            "Normalized Mutual Information (higher is better)": (
                normalized_mutual_info_score(y_true, cluster_labels)
            ),
            "Homogeneity (higher is better)": homogeneity_score(
                y_true, cluster_labels
            ),
            "Completeness (higher is better)": completeness_score(
                y_true, cluster_labels
            ),
            "V-measure (higher is better)": v_measure_score(
                y_true, cluster_labels
            ),
        }

        if name == "GMM":
            gmm = models[name]
            row["GMM AIC (lower is better)"] = gmm.aic(X_scaled)
            row["GMM BIC (lower is better)"] = gmm.bic(X_scaled)
        else:
            row["GMM AIC (lower is better)"] = np.nan
            row["GMM BIC (lower is better)"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def run_pca(X_scaled, feature_names):
    """Fit PCA, report retained variance, and calculate a 2D projection."""
    pca_full = PCA(random_state=RANDOM_STATE)
    X_pca_full = pca_full.fit_transform(X_scaled)

    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_95 = int(np.argmax(cumulative_variance >= 0.95) + 1)

    # A separate reduced representation retaining at least 95% variance.
    pca_95 = PCA(n_components=0.95, random_state=RANDOM_STATE)
    X_pca_95 = pca_95.fit_transform(X_scaled)

    # Two components are used only for 2D visualization.
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca_2d = pca_2d.fit_transform(X_scaled)

    loadings = pd.DataFrame(
        pca_full.components_.T,
        index=feature_names,
        columns=[f"PC{i + 1}" for i in range(len(feature_names))],
    )

    print("\nPCA INFORMATION")
    print("-" * 68)
    print(f"Original feature count               : {X_scaled.shape[1]}")
    print(f"Components needed for 95% variance   : {n_95}")
    print(f"Variance retained by PCA(95%)        : {pca_95.explained_variance_ratio_.sum():.4f}")
    print(
        "Variance explained by PC1 and PC2  : "
        f"{pca_2d.explained_variance_ratio_.sum():.4f}"
    )

    return (
        pca_full,
        X_pca_full,
        pca_95,
        X_pca_95,
        X_pca_2d,
        loadings,
        n_95,
    )


def save_tables(wine, X_df, y_true, labels, metrics, pca_full, loadings):
    """Save the dataset, model assignments, evaluation and PCA outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_export = X_df.copy()
    dataset_export["Reference_Class"] = [wine.target_names[i] for i in y_true]
    dataset_export.to_csv(OUTPUT_DIR / "wine_dataset_used.csv", index=False)

    assignments = X_df.copy()
    assignments["Reference_Class"] = [wine.target_names[i] for i in y_true]
    for name, cluster_labels in labels.items():
        assignments[f"{name}_Cluster"] = cluster_labels
    assignments.to_csv(OUTPUT_DIR / "cluster_assignments.csv", index=False)

    metrics.to_csv(OUTPUT_DIR / "clustering_validation_metrics.csv", index=False)
    X_df.describe().T.to_csv(OUTPUT_DIR / "descriptive_statistics.csv")
    X_df.corr().to_csv(OUTPUT_DIR / "feature_correlation_matrix.csv")
    loadings.to_csv(OUTPUT_DIR / "pca_component_loadings.csv")

    pca_variance = pd.DataFrame(
        {
            "Component": [f"PC{i + 1}" for i in range(len(pca_full.explained_variance_ratio_))],
            "Explained_Variance_Ratio": pca_full.explained_variance_ratio_,
            "Cumulative_Explained_Variance": np.cumsum(pca_full.explained_variance_ratio_),
        }
    )
    pca_variance.to_csv(OUTPUT_DIR / "pca_explained_variance.csv", index=False)


def plot_cluster_results(wine, X_pca_2d, y_true, labels, pca_full):
    """Create true-class and cluster plots using the first two PCs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_items = [("Reference classes", y_true)] + list(labels.items())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for ax, (title, plot_labels) in zip(axes, plot_items):
        scatter = ax.scatter(
            X_pca_2d[:, 0],
            X_pca_2d[:, 1],
            c=plot_labels,
            cmap="viridis",
            s=42,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.25,
        )
        ax.set_title(title)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.grid(alpha=0.2)
        fig.colorbar(scatter, ax=ax, label="Class / cluster ID")

    fig.suptitle(
        "Wine Dataset: Clusters Visualized in the First Two Principal Components\n"
        f"PC1 + PC2 explain {pca_full.explained_variance_ratio_[:2].sum():.1%} of variance",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pca_cluster_visualizations.png", dpi=180, bbox_inches="tight")


def plot_pca_variance(pca_full):
    """Plot component-wise and cumulative explained variance."""
    components = np.arange(1, len(pca_full.explained_variance_ratio_) + 1)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(components, pca_full.explained_variance_ratio_, alpha=0.75, label="Individual variance")
    ax.plot(components, cumulative, marker="o", label="Cumulative variance")
    ax.axhline(0.95, linestyle="--", linewidth=1.2, label="95% variance threshold")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA Explained Variance")
    ax.set_xticks(components)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pca_explained_variance.png", dpi=180, bbox_inches="tight")


def plot_metric_comparison(metrics):
    """Plot the main internal and external metrics for model comparison."""
    metric_specs = [
        ("Silhouette Score (higher is better)", "Silhouette score"),
        ("Davies-Bouldin Index (lower is better)", "Davies-Bouldin index"),
        ("Calinski-Harabasz Score (higher is better)", "Calinski-Harabasz score"),
        ("Adjusted Rand Index (higher is better)", "Adjusted Rand index"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()
    for ax, (column, title) in zip(axes, metric_specs):
        ax.bar(metrics["Algorithm"], metrics[column])
        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=12)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Clustering Validation Metric Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "validation_metric_comparison.png", dpi=180, bbox_inches="tight")


def plot_dendrogram(X_scaled):
    """Create a truncated Ward-linkage dendrogram for readability."""
    Z = linkage(X_scaled, method="ward")
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(
        Z,
        truncate_mode="lastp",
        p=25,
        leaf_rotation=45,
        leaf_font_size=9,
        show_contracted=True,
        ax=ax,
    )
    ax.set_title("Hierarchical Clustering Dendrogram (Ward Linkage)")
    ax.set_xlabel("Merged observations / clusters")
    ax.set_ylabel("Ward linkage distance")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hierarchical_dendrogram.png", dpi=180, bbox_inches="tight")


def print_interpretation(metrics, pca_full, n_95):
    """Print a concise, metric-aware interpretation without assuming a winner."""
    best_sil = metrics.loc[metrics["Silhouette Score (higher is better)"].idxmax(), "Algorithm"]
    lowest_db = metrics.loc[metrics["Davies-Bouldin Index (lower is better)"].idxmin(), "Algorithm"]
    best_ari = metrics.loc[metrics["Adjusted Rand Index (higher is better)"].idxmax(), "Algorithm"]

    print("\nVALIDATION SUMMARY")
    print("-" * 68)
    print(f"Highest Silhouette score : {best_sil}")
    print(f"Lowest Davies-Bouldin    : {lowest_db}")
    print(f"Highest Adjusted Rand    : {best_ari}")
    print(
        f"PCA reduced {pca_full.n_features_in_} features to {n_95} components "
        "while retaining at least 95% of the total variance."
    )
    print(
        "Interpretation: internal metrics assess compactness/separation, while "
        "external metrics compare clusters with the known dataset classes. "
        "Different metrics can favor different algorithms, so consider them together."
    )


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("K-MEANS, GMM, HIERARCHICAL CLUSTERING AND PCA")
    print("=" * 68)

    wine, X_df, X_scaled, y_true = load_and_prepare_data()
    models, labels = fit_clustering_models(X_scaled)
    metrics = calculate_validation_metrics(X_scaled, y_true, labels, models)

    (
        pca_full,
        _X_pca_full,
        _pca_95,
        _X_pca_95,
        X_pca_2d,
        loadings,
        n_95,
    ) = run_pca(X_scaled, X_df.columns)

    save_tables(wine, X_df, y_true, labels, metrics, pca_full, loadings)
    plot_cluster_results(wine, X_pca_2d, y_true, labels, pca_full)
    plot_pca_variance(pca_full)
    plot_metric_comparison(metrics)
    plot_dendrogram(X_scaled)
    print_interpretation(metrics, pca_full, n_95)

    print("\nCLUSTERING VALIDATION RESULTS")
    print("-" * 68)
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(metrics.round(4).to_string(index=False))

    print("\nOUTPUT FILES")
    print("-" * 68)
    print(f"All CSV tables and PNG charts are saved in:\n{OUTPUT_DIR}")
    print("\nAnalysis completed successfully.")

    # Display the figures after saving them. Close each plot window to continue.
    plt.show()


if __name__ == "__main__":
    main()
