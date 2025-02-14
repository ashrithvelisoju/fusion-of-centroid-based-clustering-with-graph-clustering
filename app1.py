import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import networkx as nx

# Function to perform KMeans clustering
def kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    return kmeans.labels_, kmeans.cluster_centers_

# Function to create a graph from the data
def create_graph(data, labels):
    G = nx.Graph()
    for i in range(len(data)):
        G.add_node(i, pos=(data[i][0], data[i][1]), label=labels[i])
    return G

# Function to visualize the clusters
def plot_clusters(data, labels, centers=None):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=100)
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.title('Cluster Visualization')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.grid()
    st.pyplot(plt)

# Streamlit app
st.title("Mall Customers Clustering")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('Mall_Customers.csv')
    return data

data = load_data()
st.write("Dataset Preview:")
st.dataframe(data)

# Select number of clusters
n_clusters = st.slider("Select number of clusters:", 2, 10, 5)

# Prepare data for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Perform KMeans clustering
labels, centers = kmeans_clustering(X, n_clusters)

# Create graph from clustering results
G = create_graph(X, labels)

# Plot clusters
plot_clusters(X, labels, centers)

# Display silhouette score
silhouette_avg = silhouette_score(X, labels)
st.write(f"Silhouette Score: {silhouette_avg:.2f}")

# Display graph
st.subheader("Graph Representation of Clusters")
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=True, node_color=labels, cmap='viridis', node_size=500, font_size=10)
plt.title('Graph Representation of Clusters')
st.pyplot(plt)