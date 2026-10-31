*Clustering Algorithms and PCA*
A Python machine learning project that implements and compares K-Means, Gaussian Mixture Model (GMM), and Agglomerative Hierarchical Clustering using the Scikit-learn Wine Recognition dataset. The project also applies Principal Component Analysis (PCA) for dimensionality reduction and visualization, evaluates clustering quality with multiple validation metrics, and provides an interactive Streamlit dashboard.
📌 Project Overview
The objective of this project is to explore whether unsupervised learning algorithms can identify groups of wine samples based on their chemical measurements. The three clustering algorithms are fitted using the same standardized feature data so their results can be compared under a consistent setup.
✨ Key Features
- Loads the Wine Recognition dataset directly from Scikit-learn; no manual CSV download is required.
- Preprocesses the 13 numerical features using StandardScaler.
- Implements three clustering algorithms:
  - K-Means Clustering
  - Gaussian Mixture Model (GMM)
  - Agglomerative Hierarchical Clustering
- Applies PCA to study explained variance, component loadings, and a two-dimensional projection.
- Evaluates clustering results using internal and external validation metrics.
- Generates visualizations and CSV result files in the outputs/ directory.
- Includes an interactive Streamlit dashboard to explore clustering settings and compare model results.
📊 Dataset Description
This project uses the Wine Recognition dataset included in the Scikit-learn library.
Dataset Property	Value
Total observations	178
Numerical features	13
Reference classes	3
Feature type	Chemical measurements
Dataset source	Scikit-learn load_wine()


The features include alcohol, malic acid, ash, magnesium, total phenols, flavanoids, color intensity, hue, and proline, among other chemical measurements.

Dataset loading: The dataset is loaded in Python using sklearn.datasets.load_wine(). A separate dataset download is not required.
⚠️ Important Evaluation Note
The Wine dataset includes reference class labels. These labels are not used to train the clustering models. The algorithms are fitted only on the 13 standardized numerical features. Reference labels are used after clustering only to calculate external validation metrics such as Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI).
🧠 Algorithms and Techniques
1. K-Means Clustering
K-Means partitions observations into a selected number of clusters. Each observation is assigned to its nearest cluster centroid, and the centroids are iteratively updated until the algorithm converges.
2. Gaussian Mixture Model (GMM)
GMM represents the data as a mixture of Gaussian probability distributions. Each observation is assigned to the component with the highest estimated posterior probability. The dashboard allows the user to change the covariance type and number of initializations.
3. Agglomerative Hierarchical Clustering
Agglomerative clustering begins with each observation as an individual cluster and repeatedly merges similar clusters. The batch analysis uses Ward linkage. The dashboard also allows the linkage method to be changed.
4. Principal Component Analysis (PCA)
PCA transforms the standardized input features into principal components. It is used to examine explained variance and project the observations into two dimensions for visualization. The clustering models are fitted on all 13 standardized features, not only on the two-dimensional PCA plot.
📈 Clustering Validation Metrics
The project evaluates clustering performance using the following metrics:
Metric	Purpose	Preferred Value
Silhouette Score	Measures cluster cohesion and separation.	Higher
Davies-Bouldin Index	Measures similarity between clusters.	Lower
Calinski-Harabasz Score	Compares between-cluster dispersion with within-cluster dispersion.	Higher
Adjusted Rand Index (ARI)	Measures agreement between cluster assignments and reference labels, adjusted for chance.	Higher
Normalized Mutual Information (NMI)	Measures shared information between cluster assignments and reference labels.	Higher
Homogeneity	Measures whether each cluster contains observations mainly from one reference class.	Higher
Completeness	Measures whether observations from a reference class are assigned to the same cluster.	Higher
V-measure	Combines homogeneity and completeness.	Higher


GMM information criteria: The project also reports AIC and BIC for GMM. Lower values are preferred when comparing candidate GMM fits on the same data.
🗂️ Project Structure
KMeans_GMM_Hierarchical_PCA_Project/
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── run_dashboard.bat
│
└── outputs/
    ├── wine_dataset_used.csv
    ├── cluster_assignments.csv
    ├── clustering_validation_metrics.csv
    ├── descriptive_statistics.csv
    ├── feature_correlation_matrix.csv
    ├── pca_component_loadings.csv
    ├── pca_explained_variance.csv
    ├── pca_cluster_visualizations.png
    ├── pca_explained_variance.png
    ├── validation_metric_comparison.png
    └── hierarchical_dendrogram.png
🛠️ Technologies Used
- Python – Main programming language
- NumPy – Numerical operations
- Pandas – Dataset handling and analysis
- Matplotlib – Data visualization
- SciPy – Hierarchical clustering and dendrogram support
- Scikit-learn – Preprocessing, clustering, PCA, and evaluation metrics
- Streamlit – Interactive dashboard
- Plotly – Interactive charts
⚙️ Installation and Setup
Prerequisites
- Python installed on your computer
- Git (optional, if cloning the repository)
Step 1: Clone the Repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd KMeans_GMM_Hierarchical_PCA_Project
Alternatively: Download the repository as a ZIP file and extract it.
Step 2: Create a Virtual Environment (Recommended)
Windows PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
If PowerShell blocks environment activation, you can use Command Prompt and run:
.venv\Scripts\activate.bat
Step 3: Install Dependencies
python -m pip install -r requirements.txt
▶️ Run the Project
Run the Clustering and PCA Analysis
From the project root directory, execute:
python main.py
The script performs preprocessing, runs the clustering algorithms, calculates validation metrics, performs PCA analysis, and saves charts and CSV files in the outputs/ directory. If Matplotlib figures open, close each figure window to allow the script to continue.
Launch the Interactive Streamlit Dashboard
Run:
python -m streamlit run app.py
Open the local URL shown in the terminal. The default address is usually:
http://localhost:8501
To stop the dashboard: Press Ctrl + C in the terminal.

Windows shortcut: You can also use run_dashboard.bat to launch the dashboard.
🖥️ Interactive Dashboard Sections
- Overview: Displays dataset information, model comparison, and selected validation metrics.
- Cluster Explorer: Explores cluster assignments in an interactive PCA scatter plot.
- Validation: Compares internal and external clustering validation scores.
- PCA Analysis: Displays explained variance, principal component loadings, and feature correlations.
- Dataset: Previews the dataset and provides available result-table downloads.
Interactive controls: The sidebar allows users to modify the number of clusters, random seed, GMM covariance type, GMM initializations, and hierarchical linkage method. The dashboard recalculates results based on the selected settings.
📁 Generated Output Files
After running main.py, the following files are saved in the outputs/ directory:
File Name	Description
wine_dataset_used.csv	Dataset used for the analysis.
cluster_assignments.csv	Cluster assignments from each algorithm for every observation.
clustering_validation_metrics.csv	Internal and external validation metrics and GMM information criteria.
descriptive_statistics.csv	Descriptive statistics of the numerical features.
feature_correlation_matrix.csv	Correlation matrix of the numerical features.
pca_component_loadings.csv	Feature loadings for the principal components.
pca_explained_variance.csv	Explained variance information for PCA components.
pca_cluster_visualizations.png	PCA-based views of clustering assignments and reference classes.
pca_explained_variance.png	PCA explained-variance visualization.
validation_metric_comparison.png	Comparison of clustering validation metrics.
hierarchical_dendrogram.png	Hierarchical clustering dendrogram.


🖼️ Sample Visualizations
1. PCA Clustering Visualizations

2. PCA Explained Variance

3. Clustering Validation Comparison

4. Hierarchical Clustering Dendrogram

🔍 Reproducibility and Interpretation
The batch analysis uses three clusters/components for comparison and a fixed random seed for reproducible stochastic model initialization. The interactive dashboard allows selected parameters to be changed, so its results may differ from the batch analysis when different settings are selected.

Important interpretation: A higher score on one metric does not automatically mean that an algorithm is best for every purpose. Internal metrics assess geometric cluster quality, while external metrics compare cluster assignments with the dataset's reference labels. These metrics should be interpreted together and in the context of the selected algorithm settings.
📚 References
1. Scikit-learn, Wine Recognition Dataset (load_wine) – https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html
2. Scikit-learn, Clustering – https://scikit-learn.org/stable/modules/clustering.html
3. Scikit-learn, Gaussian Mixture Models – https://scikit-learn.org/stable/modules/mixture.html
4. Scikit-learn, Principal Component Analysis (PCA) – https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
👨‍💻 Author
S G S V Ganesh
⭐ If you find this project useful, consider giving the repository a star!
