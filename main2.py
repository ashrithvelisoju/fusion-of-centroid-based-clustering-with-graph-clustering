import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

class HybridClustering:
    def __init__(self, n_clusters=3, alpha=0.5):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.kmeans = KMeans(n_clusters=n_clusters)
        
    def fit(self, X):
        # Centroid-based clustering (K-means)
        self.kmeans.fit(X)
        centroid_labels = self.kmeans.labels_
        
        # Graph-based clustering
        distances = squareform(pdist(X))
        similarity = np.exp(-distances / distances.std())
        
        G = nx.from_numpy_array(similarity)
        
        self.labels_ = centroid_labels
        return self

def load_data():
    """Load the wine quality dataset with error handling"""
    try:
        # Loading sample wine quality data
        data = {
            'fixed acidity': [7.4, 7.8, 7.8, 11.2, 7.4, 7.4, 7.9, 7.3, 7.8, 7.5],
            'volatile acidity': [0.70, 0.88, 0.76, 0.28, 0.70, 0.66, 0.60, 0.65, 0.58, 0.50],
            'citric acid': [0.00, 0.00, 0.04, 0.56, 0.00, 0.00, 0.06, 0.00, 0.02, 0.36],
            'residual sugar': [1.9, 2.6, 2.3, 1.9, 1.9, 1.8, 1.6, 1.2, 2.0, 6.1],
            'chlorides': [0.076, 0.098, 0.092, 0.075, 0.076, 0.075, 0.069, 0.065, 0.073, 0.071],
            'free sulfur dioxide': [11.0, 25.0, 15.0, 17.0, 11.0, 13.0, 15.0, 15.0, 9.0, 17.0],
            'total sulfur dioxide': [34.0, 67.0, 54.0, 60.0, 34.0, 40.0, 59.0, 21.0, 18.0, 102.0],
            'density': [0.9978, 0.9968, 0.9970, 0.9980, 0.9978, 0.9978, 0.9964, 0.9946, 0.9968, 0.9978],
            'pH': [3.51, 3.20, 3.26, 3.16, 3.51, 3.51, 3.30, 3.39, 3.36, 3.35],
            'sulphates': [0.56, 0.68, 0.65, 0.58, 0.56, 0.56, 0.46, 0.47, 0.57, 0.80],
            'alcohol': [9.4, 9.8, 9.8, 9.8, 9.4, 9.4, 9.4, 10.0, 9.5, 10.5],
            'quality': [5, 5, 5, 6, 5, 5, 5, 7, 7, 5]
        }
        df = pd.DataFrame(data)
        
        # Allow user to upload their own CSV
        uploaded_file = st.file_uploader("Upload your own wine quality dataset (CSV)", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("Custom dataset loaded successfully!")
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def preprocess_data(df):
    """Preprocess the data with error handling"""
    try:
        # Remove the target variable 'quality' for clustering
        features = df.drop('quality', axis=1)
        
        # Handle missing values
        features = features.fillna(features.mean())
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        return scaled_features, features.columns
    
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        return None, None

def main():
    st.title("Wine Quality Clustering Analysis")
    st.write("This app demonstrates hybrid clustering on the Wine Quality dataset")
    
    # Load data
    df = load_data()
    if df is None:
        st.error("Failed to load data. Please check the data source.")
        return
    
    # Display raw data
    st.subheader("Raw Data Preview")
    st.write(df.head())
    
    # Preprocess data
    scaled_features, feature_names = preprocess_data(df)
    if scaled_features is None:
        st.error("Failed to preprocess data.")
        return
    
    # Sidebar controls
    st.sidebar.header("Clustering Parameters")
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 8, 3)
    alpha = st.sidebar.slider("Hybrid Weight (α)", 0.0, 1.0, 0.5)
    
    # Perform clustering
    hybrid_clustering = HybridClustering(n_clusters=n_clusters, alpha=alpha)
    cluster_labels = hybrid_clustering.fit(scaled_features).labels_
    
    # Dimensionality reduction for visualization
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(scaled_features)
    
    # Create visualizations
    st.header("Clustering Results")
    
    # Plot 1: Scatter plot of clusters
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                        c=cluster_labels, cmap='viridis')
    ax.set_title("PCA Visualization of Clusters")
    ax.set_xlabel("First Principal Component")
    ax.set_ylabel("Second Principal Component")
    plt.colorbar(scatter)
    st.pyplot(fig)
    
    # Plot 2: Feature distribution by cluster
    st.subheader("Feature Distributions by Cluster")
    df_with_clusters = pd.DataFrame(scaled_features, columns=feature_names)
    df_with_clusters['Cluster'] = cluster_labels
    
    selected_feature = st.selectbox("Select Feature for Distribution Plot", 
                                  feature_names)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Cluster', y=selected_feature, data=df_with_clusters)
    ax.set_title(f"{selected_feature} Distribution by Cluster")
    st.pyplot(fig)
    
    # Plot 3: Cluster Centers
    st.subheader("Cluster Centers Heatmap")
    cluster_centers = pd.DataFrame(
        hybrid_clustering.kmeans.cluster_centers_,
        columns=feature_names
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(cluster_centers, annot=True, cmap='coolwarm', center=0)
    ax.set_title("Cluster Centers Heatmap")
    st.pyplot(fig)
    
    # Display cluster statistics
    st.subheader("Cluster Statistics")
    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
    st.write("Cluster Sizes:", cluster_sizes)
    
    silhouette_avg = silhouette_score(scaled_features, cluster_labels)
    st.write(f"Silhouette Score: {silhouette_avg:.3f}")

if __name__ == "__main__":
    main()