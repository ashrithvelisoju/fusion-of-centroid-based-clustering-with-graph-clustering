import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the Mall Customers dataset"""
    try:
        # Load the dataset
        data = pd.read_csv('Mall_Customers.csv')
        
        # Select features for clustering
        features = ['Annual Income (k$)', 'Spending Score (1-100)']
        X = data[features]
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, data, features
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

class HybridClustering:
    def __init__(self, n_clusters=3, alpha=0.5):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity='rbf',
            random_state=42
        )
        
    def create_similarity_graph(self, X):
        distances = squareform(pdist(X))
        sigma = np.mean(distances)
        similarities = np.exp(-distances ** 2 / (2 * sigma ** 2))
        return similarities
    
    def fit(self, X):
        # Centroid-based clustering (K-means)
        centroid_labels = self.kmeans.fit_predict(X)
        
        # Graph-based clustering using Spectral Clustering
        spectral_labels = self.spectral.fit_predict(X)
        
        # Combine both clusterings based on alpha
        mask = np.random.random(len(X)) < self.alpha
        final_labels = np.where(mask, centroid_labels, spectral_labels)
        
        self.labels_ = final_labels
        self.centroids_ = self.kmeans.cluster_centers_
        return self

def plot_clusters(X, labels, features, centroids=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    
    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], 
                  c='red', marker='x', s=200, linewidths=3,
                  label='Centroids')
        ax.legend()
    
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    plt.colorbar(scatter)
    return fig

def plot_network(X, labels, similarities, threshold=90):
    mask = similarities > np.percentile(similarities, threshold)
    G = nx.from_numpy_array(mask.astype(int))
    
    fig, ax = plt.subplots(figsize=(10, 10))
    pos = nx.spring_layout(G, k=1/np.sqrt(len(X)), iterations=50)
    nx.draw(G, pos, node_color=labels, 
            node_size=100, cmap='viridis', ax=ax)
    return fig

def main():
    st.title("Mall Customer Segmentation using Hybrid Clustering")
    st.write("""
    This application demonstrates customer segmentation using a hybrid approach that combines
    centroid-based (K-means) and graph-based (Spectral) clustering techniques.
    """)
    
    # Load and preprocess data
    X, data, features = load_and_preprocess_data()
    if X is None or data is None:
        return
    
    # Sidebar controls
    st.sidebar.header("Clustering Parameters")
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 8, 5)
    alpha = st.sidebar.slider(
        "Centroid-Graph Fusion Weight (α)", 
        0.0, 1.0, 0.5,
        help="0: Pure graph-based, 1: Pure centroid-based"
    )
    
    # Perform clustering
    with st.spinner("Performing clustering analysis..."):
        try:
            hybrid_clustering = HybridClustering(n_clusters=n_clusters, alpha=alpha)
            hybrid_clustering.fit(X)
            similarities = hybrid_clustering.create_similarity_graph(X)
        except Exception as e:
            st.error(f"Error during clustering: {str(e)}")
            return
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Cluster Visualization", "Customer Statistics", "Network Analysis"])
    
    with tab1:
        st.subheader("Customer Segments Visualization")
        fig = plot_clusters(X, hybrid_clustering.labels_, features, hybrid_clustering.centroids_)
        st.pyplot(fig)
        
        # Add cluster interpretation
        cluster_sizes = pd.Series(hybrid_clustering.labels_).value_counts()
        st.write("### Cluster Sizes")
        st.write(pd.DataFrame({
            'Cluster': cluster_sizes.index,
            'Number of Customers': cluster_sizes.values
        }))
    
    with tab2:
        st.subheader("Customer Segment Statistics")
        data['Cluster'] = hybrid_clustering.labels_
        cluster_stats = data.groupby('Cluster').agg({
            'Annual Income (k$)': ['mean', 'std'],
            'Spending Score (1-100)': ['mean', 'std']
        }).round(2)
        
        st.write("### Detailed Statistics by Cluster")
        st.write(cluster_stats)
        
        # Add visualization of cluster characteristics
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=data, x='Cluster', y='Annual Income (k$)')
        st.pyplot(fig)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=data, x='Cluster', y='Spending Score (1-100)')
        st.pyplot(fig)
    
    with tab3:
        st.subheader("Customer Network Analysis")
        fig = plot_network(X, hybrid_clustering.labels_, similarities)
        st.pyplot(fig)
    
    # Display clustering quality
    silhouette_avg = silhouette_score(X, hybrid_clustering.labels_)
    st.metric("Clustering Quality (Silhouette Score)", f"{silhouette_avg:.3f}")

if __name__ == "__main__":
    main()