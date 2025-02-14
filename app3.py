import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

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
        self.spectral_labels_ = spectral_labels  # Store spectral labels separately
        return self

# New graph creation function
def create_graph(data, cluster_assignments):
    G = nx.Graph()
    for _, row in data.iterrows():
        G.add_node(row['CustomerID'], pos=(row['Annual Income (k$)'], row['Spending Score (1-100)']))
    
    for i, row1 in data.iterrows():
        for j, row2 in data.iterrows():
            if i < j:  # Avoid duplicate edges
                dist = np.linalg.norm(np.array(row1[['Annual Income (k$)', 'Spending Score (1-100)']]) - 
                                      np.array(row2[['Annual Income (k$)', 'Spending Score (1-100)']]))
                if cluster_assignments[i] == cluster_assignments[j]:  # Connect nodes within the same cluster
                    G.add_edge(row1['CustomerID'], row2['CustomerID'], weight=dist)
    
    return G

@st.cache_data
def load_mall_customers_data():
    try:
        df = pd.read_csv('Mall_Customers.csv')
        return df
    except FileNotFoundError:
        st.error("Please ensure 'Mall_Customers.csv' is in the same directory as the script.")
        return None

def preprocess_data(df):
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, features

def plot_clusters(X_scaled, labels, features, centroids=None, title="Clustering Results"):
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                        c=labels, cmap='viridis')
    
    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], 
                  c='red', marker='x', s=200, linewidths=3,
                  label='Centroids')
        ax.legend()
    
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_title(title)
    plt.colorbar(scatter)
    return fig

def plot_network(X_scaled, labels, similarities, threshold=90, title="Network Graph"):
    mask = similarities > np.percentile(similarities, threshold)
    G = nx.from_numpy_array(mask.astype(int))
    
    fig, ax = plt.subplots(figsize=(10, 10))
    pos = nx.spring_layout(G, k=1/np.sqrt(len(X_scaled)), iterations=50)
    nx.draw(G, pos, node_color=labels, 
            node_size=100, cmap='viridis', ax=ax)
    ax.set_title(title)
    return fig

def plot_custom_graph(G, labels, title="Custom Graph View"):
    fig, ax = plt.subplots(figsize=(10, 10))
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, node_color=labels, 
            node_size=100, cmap='viridis', 
            edge_color='gray', ax=ax)
    ax.set_title(title)
    return fig

def main():
    st.title("Hybrid Clustering Analysis: Mall Customer Segmentation")
    st.write("""
    This app demonstrates the fusion of centroid-based (K-means) and graph-based clustering
    techniques for customer segmentation analysis.
    """)
    
    # Load data
    df = load_mall_customers_data()
    if df is None:
        return
    
    # Preprocess data
    X_scaled, features = preprocess_data(df)
    
    # Sidebar controls
    st.sidebar.header("Clustering Parameters")
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 8, 5)
    alpha = st.sidebar.slider("Centroid-Graph Fusion Weight (α)", 0.0, 1.0, 0.5,
                            help="0: Pure graph-based, 1: Pure centroid-based")
    
    # Perform clustering
    with st.spinner("Performing clustering analysis..."):
        try:
            hybrid_clustering = HybridClustering(n_clusters=n_clusters, alpha=alpha)
            hybrid_clustering.fit(X_scaled)
            similarities = hybrid_clustering.create_similarity_graph(X_scaled)
            
            # Create custom graph
            custom_graph = create_graph(df, hybrid_clustering.labels_)
            spectral_custom_graph = create_graph(df, hybrid_clustering.spectral_labels_)
        except Exception as e:
            st.error(f"Error during clustering: {str(e)}")
            return
    
    # Visualizations
    st.header("Clustering Results")
    
    # Add new tab for Custom Graph View
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Hybrid Scatter Plot", "Cluster Statistics", 
                                          "Hybrid Network Graph", "Graph Clustering View",
                                          "Custom Graph View"])
    
    with tab1:
        fig = plot_clusters(X_scaled, hybrid_clustering.labels_, 
                          features, hybrid_clustering.centroids_,
                          title="Hybrid Clustering Results")
        st.pyplot(fig)
        
    with tab2:
        df['Cluster'] = hybrid_clustering.labels_
        cluster_stats = df.groupby('Cluster').agg({
            'Annual Income (k$)': ['mean', 'std'],
            'Spending Score (1-100)': ['mean', 'std']
        }).round(2)
        st.write("Cluster Statistics:")
        st.write(cluster_stats)
        
    with tab3:
        fig = plot_network(X_scaled, hybrid_clustering.labels_, similarities,
                         title="Hybrid Network Graph")
        st.pyplot(fig)
    
    with tab4:
        # Graph Clustering View (Spectral Clustering results)
        st.subheader("Graph-Based Clustering (Spectral Clustering)")
        # Scatter plot for spectral clustering
        fig_spectral = plot_clusters(X_scaled, hybrid_clustering.spectral_labels_,
                                  features, title="Spectral Clustering Results")
        st.pyplot(fig_spectral)
        
        # Network graph for spectral clustering
        fig_spectral_network = plot_network(X_scaled, hybrid_clustering.spectral_labels_,
                                         similarities, title="Spectral Clustering Network Graph")
        st.pyplot(fig_spectral_network)
    
    with tab5:
        # Custom Graph View
        st.subheader("Custom Graph-Based Views")
        # Hybrid clustering custom graph
        st.write("Hybrid Clustering Custom Graph")
        fig_custom_hybrid = plot_custom_graph(custom_graph, hybrid_clustering.labels_,
                                           title="Hybrid Clustering Custom Graph")
        st.pyplot(fig_custom_hybrid)
        
        # Spectral clustering custom graph
        st.write("Spectral Clustering Custom Graph")
        fig_custom_spectral = plot_custom_graph(spectral_custom_graph, 
                                             hybrid_clustering.spectral_labels_,
                                             title="Spectral Clustering Custom Graph")
        st.pyplot(fig_custom_spectral)
    
    # Performance metrics
    silhouette_avg = silhouette_score(X_scaled, hybrid_clustering.labels_)
    spectral_silhouette = silhouette_score(X_scaled, hybrid_clustering.spectral_labels_)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Hybrid Clustering Quality (Silhouette Score)", f"{silhouette_avg:.3f}")
    with col2:
        st.metric("Spectral Clustering Quality (Silhouette Score)", f"{spectral_silhouette:.3f}")

if __name__ == "__main__":
    main()