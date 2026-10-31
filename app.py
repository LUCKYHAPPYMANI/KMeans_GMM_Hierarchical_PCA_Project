"""Interactive Streamlit dashboard for clustering and PCA analysis."""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
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


st.set_page_config(
    page_title="Clustering & PCA Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data(show_spinner=False)
def load_dataset():
    """Load the built-in Wine dataset and standardize its numeric features."""
    wine = load_wine(as_frame=True)
    X = wine.data.copy()
    y = wine.target.to_numpy()
    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, y, list(wine.target_names), list(X.columns)


def fit_models(X_scaled, n_clusters, random_state, gmm_covariance, gmm_n_init, linkage_method):
    """Fit all clustering methods using features only (never the reference labels)."""
    model_objects = {
        "K-Means": KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=20,
        ),
        "GMM": GaussianMixture(
            n_components=n_clusters,
            covariance_type=gmm_covariance,
            random_state=random_state,
            n_init=gmm_n_init,
        ),
        "Hierarchical": AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method,
        ),
    }
    predicted = {
        name: model.fit_predict(X_scaled)
        for name, model in model_objects.items()
    }
    return model_objects, predicted


def validation_table(X_scaled, y_true, predictions, models):
    """Calculate internal and external validation metrics after fitting."""
    rows = []
    for name, cluster_ids in predictions.items():
        row = {
            "Algorithm": name,
            "Silhouette Score": silhouette_score(X_scaled, cluster_ids),
            "Davies-Bouldin Index": davies_bouldin_score(X_scaled, cluster_ids),
            "Calinski-Harabasz Score": calinski_harabasz_score(X_scaled, cluster_ids),
            "Adjusted Rand Index": adjusted_rand_score(y_true, cluster_ids),
            "Normalized Mutual Information": normalized_mutual_info_score(y_true, cluster_ids),
            "Homogeneity": homogeneity_score(y_true, cluster_ids),
            "Completeness": completeness_score(y_true, cluster_ids),
            "V-measure": v_measure_score(y_true, cluster_ids),
        }
        if name == "GMM":
            row["GMM AIC"] = models[name].aic(X_scaled)
            row["GMM BIC"] = models[name].bic(X_scaled)
        else:
            row["GMM AIC"] = np.nan
            row["GMM BIC"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def make_download_frame(X, y_true, class_names, predictions):
    """Create a downloadable table containing features and cluster assignments."""
    result = X.copy()
    result.insert(0, "Reference_Class", [class_names[i] for i in y_true])
    for name, cluster_ids in predictions.items():
        result[f"{name}_Cluster"] = cluster_ids
    return result


def metric_bar_chart(metrics, metric, higher_is_better=True):
    ordered = metrics.sort_values(metric, ascending=not higher_is_better)
    fig = px.bar(
        ordered,
        x="Algorithm",
        y=metric,
        color="Algorithm",
        text_auto=".3f",
        title=metric,
    )
    fig.update_layout(showlegend=False, height=360, margin=dict(l=20, r=20, t=55, b=20))
    return fig


def render_cluster_scatter(X_pca, cluster_ids, reference_ids, class_names, label_mode, algorithm):
    plot_df = pd.DataFrame(
        {
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Cluster": [f"Cluster {v}" for v in cluster_ids],
            "Reference class": [class_names[v] for v in reference_ids],
            "Observation": np.arange(1, len(cluster_ids) + 1),
        }
    )
    if label_mode == "Reference class":
        color_col = "Reference class"
        title = f"{algorithm}: reference classes in PCA space"
    else:
        color_col = "Cluster"
        title = f"{algorithm}: cluster assignments in PCA space"

    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color=color_col,
        hover_data=["Observation", "Cluster", "Reference class"],
        title=title,
        opacity=0.85,
    )
    fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="white")))
    fig.update_layout(height=560, legend_title_text=color_col)
    return fig


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    st.title("🧪 Clustering Algorithms & PCA Dashboard")
    st.caption("Explore K-Means, Gaussian Mixture Model (GMM), Hierarchical Clustering and PCA interactively.")

    X, X_scaled, y_true, class_names, feature_names = load_dataset()

    st.sidebar.header("Analysis controls")
    st.sidebar.caption("Change the settings and the dashboard will recompute the clustering results.")
    n_clusters = st.sidebar.slider("Number of clusters", min_value=2, max_value=8, value=3)
    random_state = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
    gmm_covariance = st.sidebar.selectbox("GMM covariance type", ["full", "tied", "diag", "spherical"])
    gmm_n_init = st.sidebar.slider("GMM initializations", min_value=1, max_value=15, value=5)
    linkage_method = st.sidebar.selectbox("Hierarchical linkage", ["ward", "complete", "average", "single"])
    st.sidebar.divider()
    st.sidebar.info(
        "Reference Wine classes are not used to train the clustering models. "
        "They are used only for external validation after clustering."
    )

    if linkage_method == "ward":
        st.sidebar.caption("Ward linkage is restricted to Euclidean distance and tends to form compact clusters.")

    with st.spinner("Fitting clustering models and calculating validation metrics..."):
        models, predictions = fit_models(
            X_scaled,
            n_clusters,
            int(random_state),
            gmm_covariance,
            gmm_n_init,
            linkage_method,
        )
        metrics = validation_table(X_scaled, y_true, predictions, models)
        pca = PCA(n_components=2, random_state=int(random_state))
        X_pca = pca.fit_transform(X_scaled)
        pca_full = PCA(random_state=int(random_state)).fit(X_scaled)
        variance_95 = int(np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.95) + 1)

    best_silhouette = metrics.loc[metrics["Silhouette Score"].idxmax(), "Algorithm"]
    best_ari = metrics.loc[metrics["Adjusted Rand Index"].idxmax(), "Algorithm"]
    best_db = metrics.loc[metrics["Davies-Bouldin Index"].idxmin(), "Algorithm"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", f"{len(X):,}")
    c2.metric("Numeric features", len(feature_names))
    c3.metric("Best silhouette", best_silhouette)
    c4.metric("PCA components for 95% variance", variance_95)

    tab_overview, tab_explorer, tab_validation, tab_pca, tab_data = st.tabs(
        ["Overview", "Cluster Explorer", "Validation", "PCA Analysis", "Dataset"]
    )

    with tab_overview:
        st.subheader("Project overview")
        st.write(
            "The Wine recognition dataset contains 178 observations, 13 numeric chemical measurements "
            "and 3 reference classes. Features are standardized before clustering because they use "
            "different measurement scales."
        )
        left, right = st.columns(2)
        with left:
            st.plotly_chart(metric_bar_chart(metrics, "Silhouette Score", True), use_container_width=True)
        with right:
            st.plotly_chart(metric_bar_chart(metrics, "Adjusted Rand Index", True), use_container_width=True)
        st.subheader("Current model comparison")
        st.dataframe(metrics.round(4), use_container_width=True, hide_index=True)
        st.caption(
            f"Current leaders: silhouette — {best_silhouette}; Davies-Bouldin — {best_db}; "
            f"Adjusted Rand Index — {best_ari}. A higher silhouette/ARI and lower Davies-Bouldin are preferred."
        )

    with tab_explorer:
        st.subheader("Explore cluster assignments")
        algorithm = st.selectbox("Select clustering algorithm", list(predictions.keys()))
        label_mode = st.radio(
            "Color points by",
            ["Cluster assignment", "Reference class"],
            horizontal=True,
        )
        st.plotly_chart(
            render_cluster_scatter(
                X_pca,
                predictions[algorithm],
                y_true,
                class_names,
                label_mode,
                algorithm,
            ),
            use_container_width=True,
        )
        st.caption(
            f"The first two principal components explain {pca.explained_variance_ratio_.sum():.1%} "
            "of total standardized-feature variance. PCA is used here for visualization; clustering "
            "was performed on all 13 standardized features."
        )
        cluster_counts = pd.Series(predictions[algorithm]).value_counts().sort_index()
        count_df = pd.DataFrame({"Cluster": [f"Cluster {i}" for i in cluster_counts.index], "Observations": cluster_counts.values})
        st.plotly_chart(
            px.bar(count_df, x="Cluster", y="Observations", color="Cluster", title=f"{algorithm}: observations per cluster", text_auto=True).update_layout(showlegend=False, height=350),
            use_container_width=True,
        )
        if algorithm == "Hierarchical":
            st.markdown("**Hierarchical clustering dendrogram (Ward / selected linkage)**")
            linkage_matrix = linkage(X_scaled, method=linkage_method)
            dendro_fig, dendro_ax = plt.subplots(figsize=(12, 5))
            dendrogram(
                linkage_matrix,
                truncate_mode="lastp",
                p=25,
                leaf_rotation=45,
                leaf_font_size=9,
                show_contracted=True,
                ax=dendro_ax,
            )
            dendro_ax.set_xlabel("Merged observations / clusters")
            dendro_ax.set_ylabel("Linkage distance")
            dendro_ax.set_title("Hierarchical clustering dendrogram")
            dendro_fig.tight_layout()
            st.pyplot(dendro_fig, use_container_width=True)
            plt.close(dendro_fig)

    with tab_validation:
        st.subheader("Clustering validation")
        st.markdown(
            "**Internal validation** evaluates the geometry of the clusters without true class labels. "
            "**External validation** compares the final cluster assignments with the dataset's reference classes."
        )
        metric_help = {
            "Silhouette Score": "Higher is better. Measures how close each point is to its own cluster compared with other clusters.",
            "Davies-Bouldin Index": "Lower is better. Measures similarity between clusters using within-cluster dispersion and separation.",
            "Calinski-Harabasz Score": "Higher is generally better. Compares between-cluster and within-cluster dispersion.",
            "Adjusted Rand Index": "Higher is better. Measures agreement with reference classes, adjusted for chance.",
            "Normalized Mutual Information": "Higher is better. Measures shared information between clusters and reference classes.",
        }
        metric_choice = st.selectbox("Choose a metric to compare", list(metric_help.keys()))
        st.info(metric_help[metric_choice])
        higher = metric_choice != "Davies-Bouldin Index"
        st.plotly_chart(metric_bar_chart(metrics, metric_choice, higher), use_container_width=True)
        st.dataframe(metrics.round(5), use_container_width=True, hide_index=True)
        st.download_button(
            "Download validation metrics (CSV)",
            data=metrics.to_csv(index=False).encode("utf-8"),
            file_name="clustering_validation_metrics.csv",
            mime="text/csv",
        )

        gmm_model = models["GMM"]
        st.caption(f"GMM information criteria for the current configuration — AIC: {gmm_model.aic(X_scaled):,.2f}; BIC: {gmm_model.bic(X_scaled):,.2f} (lower is preferred when comparing GMM fits on the same data).")

    with tab_pca:
        st.subheader("Principal Component Analysis")
        pca_left, pca_right = st.columns(2)
        with pca_left:
            st.metric("Variance explained by PC1 + PC2", f"{pca.explained_variance_ratio_.sum():.2%}")
        with pca_right:
            st.metric("Components required for at least 95% variance", variance_95)
        variance_df = pd.DataFrame(
            {
                "Component": [f"PC{i + 1}" for i in range(len(pca_full.explained_variance_ratio_))],
                "Individual variance": pca_full.explained_variance_ratio_,
                "Cumulative variance": np.cumsum(pca_full.explained_variance_ratio_),
            }
        )
        variance_long = variance_df.melt(id_vars="Component", var_name="Measure", value_name="Explained variance")
        fig = px.bar(
            variance_long[variance_long["Measure"] == "Individual variance"],
            x="Component",
            y="Explained variance",
            title="Variance explained by each principal component",
            text_auto=".1%",
        )
        fig.add_scatter(x=variance_df["Component"], y=variance_df["Cumulative variance"], mode="lines+markers", name="Cumulative variance", yaxis="y")
        fig.add_hline(y=0.95, line_dash="dash", annotation_text="95% threshold")
        fig.update_layout(height=430, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**PCA component loadings**")
        loadings = pd.DataFrame(
            pca_full.components_.T,
            index=feature_names,
            columns=[f"PC{i + 1}" for i in range(len(feature_names))],
        )
        st.dataframe(loadings.iloc[:, :5].round(4), use_container_width=True)
        st.download_button(
            "Download PCA loadings (CSV)",
            data=loadings.to_csv().encode("utf-8"),
            file_name="pca_component_loadings.csv",
            mime="text/csv",
        )

        st.markdown("**Feature correlation heatmap**")
        corr = X.corr()
        heatmap = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlation between original numeric features",
        )
        heatmap.update_layout(height=700)
        st.plotly_chart(heatmap, use_container_width=True)

    with tab_data:
        st.subheader("Dataset preview and cluster assignments")
        st.write("The reference class is shown for evaluation and interpretation only; it is not used as an input to clustering.")
        full_result = make_download_frame(X, y_true, class_names, predictions)
        st.dataframe(full_result.head(30), use_container_width=True, hide_index=True)
        st.download_button(
            "Download all observations and cluster assignments (CSV)",
            data=full_result.to_csv(index=False).encode("utf-8"),
            file_name="cluster_assignments.csv",
            mime="text/csv",
        )
        st.markdown("**Descriptive statistics**")
        st.dataframe(X.describe().T.round(3), use_container_width=True)

    st.divider()
    st.caption("Dataset: scikit-learn Wine recognition dataset. Unsupervised algorithms are trained without the reference class labels.")


if __name__ == "__main__":
    main()
