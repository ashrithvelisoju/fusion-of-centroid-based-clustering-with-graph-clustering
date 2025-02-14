import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv('Mall_Customers.csv')
    return data

# Function to compute pairwise dissimilarities (Euclidean distances squared)
def compute_dissimilarities(X):
    n_samples = X.shape[0]
    W = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            if i != j:
                W[i, j] = np.linalg.norm(X[i] - X[j])**2
    return W

# Function to compute weighted dissimilarity (Δ_i^k)
def compute_weighted_dissimilarity(X, W, cluster_assignments, k):
    n_samples = X.shape[0]
    delta = np.zeros(n_samples)
    for i in range(n_samples):
        cluster_k_indices = np.where(cluster_assignments == k)[0]
        delta[i] = np.sum(W[i, cluster_k_indices])
    return delta

# Hybrid clustering with weighted dissimilarity loss
def hybrid_clustering(X, n_clusters, alpha=0.5, max_iter=100, tol=1e-4):
    n_samples, n_features = X.shape
    
    # Initialize with KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    cluster_assignments = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    # Compute pairwise dissimilarities
    W = compute_dissimilarities(X)
    
    # EM-like iteration
    prev_cost = float('inf')
    for iteration in range(max_iter):
        # E-step: Compute weighted dissimilarities and hybrid dissimilarities
        new_assignments = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            hybrid_dissimilarities = []
            for k in range(n_clusters):
                # Centroid-based distance (squared Euclidean)
                d_i_k = np.linalg.norm(X[i] - centroids[k])**2
                
                # Weighted dissimilarity (Δ_i^k)
                delta_i_k = compute_weighted_dissimilarity(X, W, cluster_assignments, k)[i]
                
                # Hybrid dissimilarity (Equation 15 from paper)
                tau_i_k = (1 - alpha) * d_i_k + alpha * delta_i_k
                hybrid_dissimilarities.append(tau_i_k)
            
            new_assignments[i] = np.argmin(hybrid_dissimilarities)
        
        # M-step: Update centroids
        cluster_assignments = new_assignments
        for k in range(n_clusters):
            cluster_points = X[cluster_assignments == k]
            if len(cluster_points) > 0:
                centroids[k] = np.mean(cluster_points, axis=0)
        
        # Compute cost function (simplified version of Q_H from paper)
        cost = 0
        for i in range(n_samples):
            k = cluster_assignments[i]
            d_i_k = np.linalg.norm(X[i] - centroids[k])**2
            delta_i_k = compute_weighted_dissimilarity(X, W, cluster_assignments, k)[i]
            cost += (1 - alpha) * d_i_k + alpha * delta_i_k
        
        # Check convergence
        if abs(prev_cost - cost) < tol:
            break
        prev_cost = cost
    
    return cluster_assignments, centroids

# Function to create a graph from data
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

# Clustering and visualization
def cluster_and_visualize(data):
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    X = data[features].values
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Parameters
    n_clusters = st.slider('Select number of clusters:', 2, 10, 5)
    alpha = st.slider('Select alpha (weight for graph clustering):', 0.0, 1.0, 0.5)
    
    # Apply hybrid clustering
    cluster_assignments, centroids = hybrid_clustering(X_scaled, n_clusters, alpha)
    data['Cluster'] = cluster_assignments
    
    # Scatter plot for centroid-based clustering
    fig, ax = plt.subplots()
    scatter = ax.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], 
                         c=data['Cluster'], cmap='viridis')
    ax.set_title('Hybrid Clustering (Centroid + Graph)')
    ax.set_xlabel('Annual Income (k$)')
    ax.set_ylabel('Spending Score (1-100)')
    plt.colorbar(scatter)
    st.pyplot(fig)
    
    # Graph for graph clustering
    G = create_graph(data, cluster_assignments)
    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, node_color=data['Cluster'], cmap='viridis', with_labels=False, node_size=50, edge_color='gray')
    plt.title('Graph Clustering View')
    st.pyplot(plt)

# Main function
def main():
    st.title('Fusion of Centroid-Based Clustering with Graph Clustering')
    data = load_data()
    st.write("Dataset Preview:")
    st.write(data.head())
    
    cluster_and_visualize(data)

if __name__ == '__main__':
    main()