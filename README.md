# Fusion of Centroid-Based Clustering with Graph Clustering

A comprehensive machine learning application that combines centroid-based (K-means) and graph-based (Spectral) clustering techniques for advanced customer segmentation and data analysis.

## 📋 Table of Contents
- [Overview and Purpose](#overview-and-purpose)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation Instructions](#installation-instructions)
- [Configuration Guide](#configuration-guide)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Contribution Guidelines](#contribution-guidelines)
- [License Information](#license-information)

## 🎯 Overview and Purpose

This project implements a novel hybrid clustering approach that combines the strengths of both centroid-based and graph-based clustering algorithms. The application provides multiple interactive Streamlit interfaces for analyzing customer data and wine quality datasets, focusing on advanced segmentation techniques using annual income, spending scores, and wine quality features.

### Core Methodology
The hybrid approach leverages:
- **Centroid-based clustering (K-means)**: Efficient geometric clustering based on distance to cluster centers
- **Graph-based clustering (Spectral)**: Connectivity-aware clustering using similarity graphs
- **Weighted fusion mechanism**: Configurable parameter α to control the balance between approaches

The fusion methodology allows for more robust clustering results by combining:
- The geometric properties and computational efficiency of K-means
- The connectivity patterns and complex data structure handling of spectral clustering

## ✨ Features

### Core Clustering Capabilities
- **Hybrid Clustering Algorithm**: Seamlessly combines K-means and Spectral clustering with configurable fusion weights (α parameter)
- **Multiple Implementation Approaches**:
  - Simple hybrid fusion (`app.py`, `app3.py`)
  - Advanced weighted dissimilarity-based fusion (`app2.py`)
  - EM-style iterative optimization for optimal cluster assignments
- **Flexible Parameter Control**: Real-time adjustment of cluster count (2-8) and fusion weight (0.0-1.0)

### Comprehensive Visualizations
- **Interactive Scatter Plots**: Cluster assignments with centroid visualization
- **Network Graph Representations**: Similarity-based connectivity graphs
- **Custom Graph Views**: Customer-specific graph networks with intra-cluster connections
- **Statistical Analysis**: Detailed cluster statistics and performance metrics
- **Multi-tab Interface**: Organized visualization across 5 specialized tabs

### Dataset Support
- **Mall Customer Segmentation**: Customer ID, demographics, income, and spending analysis
- **Wine Quality Analysis**: Multi-dimensional wine feature clustering
- **Extensible Architecture**: Easy adaptation for additional datasets

### Performance Evaluation
- **Silhouette Score Analysis**: Automated clustering quality assessment
- **Comparative Metrics**: Side-by-side evaluation of hybrid vs. pure spectral clustering
- **Real-time Performance Monitoring**: Live updates as parameters change

## 📋 Prerequisites

### System Requirements
- **Operating System**: Windows 10/11, macOS 10.14+, or Linux Ubuntu 18.04+
- **Python**: Version 3.7 or higher (Python 3.8+ recommended)
- **Memory**: Minimum 8GB RAM (16GB recommended for larger datasets)
- **Storage**: At least 1GB free disk space
- **Network**: Internet connection for initial package installation

### Software Dependencies
- **Package Manager**: pip (included with Python)
- **Web Browser**: Modern browser (Chrome, Firefox, Safari, Edge) for Streamlit interface
- **Git**: For version control and repository cloning (optional)

## 🚀 Installation Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/ashrithvelisoju/fusion-of-centroid-based-clustering-with-graph-clustering.git
cd "fusion of centroid-based clustering with graph clustering"
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv clustering_env

# Activate virtual environment
# On Windows:
clustering_env\Scripts\activate
# On macOS/Linux:
source clustering_env/bin/activate
```

### 3. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Alternative: Install packages individually
pip install streamlit pandas numpy scikit-learn networkx matplotlib seaborn
```

### 4. Verify Installation
```bash
# Test all dependencies
python -c "import streamlit, pandas, sklearn, networkx, matplotlib, seaborn; print('✅ All dependencies installed successfully')"

# Check Streamlit version
streamlit --version
```

### 5. Validate Data Files
Ensure the following CSV files are present in the project root:
- `Mall_Customers.csv` (200+ customer records)
- `winequality-red.csv` (wine quality dataset)

## ⚙️ Configuration Guide

### Dataset Configuration
The application automatically detects and loads datasets from the project directory:

```python
# Mall Customers Dataset Structure
CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)
1, Male, 19, 15, 39
2, Male, 21, 15, 81
...
```

### Application Parameters

#### Clustering Parameters
- **Number of Clusters**: 2-8 clusters (default: 5)
  - Optimal range typically 3-6 for customer segmentation
  - Use silhouette score to guide selection
  
- **Alpha (α) Fusion Parameter**: 0.0-1.0 (default: 0.5)
  - `α = 0.0`: Pure graph-based (spectral) clustering
  - `α = 0.5`: Balanced hybrid approach
  - `α = 1.0`: Pure centroid-based (K-means) clustering

#### Advanced Configuration (app2.py)
- **Maximum Iterations**: 100 (default for EM-style optimization)
- **Convergence Tolerance**: 1e-4 (cost function convergence threshold)
- **Similarity Threshold**: 90th percentile (for network graph visualization)

### Environment Variables
```bash
# Optional: Set Streamlit configuration
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_HEADLESS=true
```

## 💻 Usage Examples

### Quick Start - Enhanced Customer Segmentation
```bash
# Run the main enhanced application
streamlit run app3.py
```
This launches the most comprehensive interface with all visualization tabs and features.

### Application-Specific Usage

#### 1. Basic Hybrid Clustering
```bash
streamlit run app.py
```
- Simple hybrid implementation
- Basic visualization capabilities
- Good for initial exploration

#### 2. Advanced Weighted Dissimilarity Approach
```bash
streamlit run app2.py
```
- EM-style iterative optimization
- Custom dissimilarity matrix computation
- Research-grade implementation

#### 3. Mall Customer Analysis
```bash
streamlit run main1.py
```
- Specialized customer segmentation interface
- Enhanced preprocessing and visualization
- Customer-focused analytics

#### 4. Wine Quality Clustering
```bash
streamlit run main2.py
```
- Multi-dimensional wine feature analysis
- PCA-based dimensionality reduction
- Quality-based clustering insights

### Programmatic Usage

#### Basic HybridClustering Implementation
```python
from app3 import HybridClustering
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
data = pd.read_csv('Mall_Customers.csv')
features = ['Annual Income (k$)', 'Spending Score (1-100)']
X = data[features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and fit hybrid clustering
hybrid_clustering = HybridClustering(n_clusters=5, alpha=0.5)
hybrid_clustering.fit(X_scaled)

# Access results
labels = hybrid_clustering.labels_
centroids = hybrid_clustering.centroids_
spectral_labels = hybrid_clustering.spectral_labels_

print(f"Cluster assignments: {labels}")
print(f"Number of clusters found: {len(np.unique(labels))}")
```

#### Advanced Parameter Optimization
```python
from sklearn.metrics import silhouette_score
import numpy as np

# Parameter grid search
alpha_values = np.linspace(0.0, 1.0, 11)
cluster_ranges = range(2, 9)
best_score = -1
best_params = {}

for n_clusters in cluster_ranges:
    for alpha in alpha_values:
        clustering = HybridClustering(n_clusters=n_clusters, alpha=alpha)
        clustering.fit(X_scaled)
        score = silhouette_score(X_scaled, clustering.labels_)
        
        if score > best_score:
            best_score = score
            best_params = {'n_clusters': n_clusters, 'alpha': alpha}
            
print(f"Best parameters: {best_params}")
print(f"Best silhouette score: {best_score:.3f}")
```

#### Custom Graph Analysis
```python
from app3 import create_graph
import networkx as nx

# Create custom graph from clustering results
custom_graph = create_graph(data, hybrid_clustering.labels_)

# Analyze graph properties
print(f"Number of nodes: {custom_graph.number_of_nodes()}")
print(f"Number of edges: {custom_graph.number_of_edges()}")
print(f"Graph density: {nx.density(custom_graph):.3f}")

# Find connected components
components = list(nx.connected_components(custom_graph))
print(f"Connected components: {len(components)}")
```

### Integration with External Data
```python
# Adapt for custom datasets
def process_custom_data(csv_file, feature_columns):
    """Process any CSV dataset for hybrid clustering"""
    data = pd.read_csv(csv_file)
    X = data[feature_columns]
    
    # Handle missing values
    X = X.dropna()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, data, scaler

# Example usage
X_scaled, data, scaler = process_custom_data('your_data.csv', ['feature1', 'feature2'])
clustering = HybridClustering(n_clusters=4, alpha=0.3)
clustering.fit(X_scaled)
```

## 📁 Project Structure

```
fusion-of-centroid-based-clustering-with-graph-clustering/
├── 📄 README.md                     # This comprehensive documentation
├── 📄 requirements.txt              # Python dependencies
├── 📊 Mall_Customers.csv           # Customer segmentation dataset (200+ records)
├── 📊 winequality-red.csv          # Wine quality dataset
│
├── 🎯 Core Applications
│   ├── 📱 app3.py                  # ⭐ Enhanced clustering (RECOMMENDED)
│   ├── 📱 app.py                   # Basic hybrid implementation
│   ├── 📱 app2.py                  # Advanced weighted dissimilarity approach
│   ├── 📱 main1.py                 # Mall customer specialized interface
│   └── 📱 main2.py                 # Wine quality specialized interface
│
├── 🔧 Standalone Scripts
│   ├── 📱 app1.py                  # Simple K-means clustering
│   └── 📱 main.py                  # Basic wine quality K-means
│
└── 🗂️ Environment
    ├── 📁 .venv/                   # Virtual environment (if created)
    └── 📁 .git/                    # Git repository metadata
```

### Key Components Description

#### Core Classes and Functions

**HybridClustering Class** (`app3.py`, `main1.py`)
- Main clustering algorithm implementation
- Combines K-means and Spectral clustering
- Configurable fusion parameter (α)
- Methods: `__init__()`, `create_similarity_graph()`, `fit()`

**Advanced Functions** (`app2.py`)
- `compute_dissimilarities()`: Pairwise Euclidean distance calculation
- `compute_weighted_dissimilarity()`: Weighted cluster dissimilarity computation
- `hybrid_clustering()`: EM-style iterative optimization

**Visualization Functions**
- `plot_clusters()`: Scatter plot with centroid visualization
- `plot_network()`: Network graph representation
- `plot_custom_graph()`: Customer-specific graph networks
- `create_graph()`: Graph construction from clustering results

**Data Processing**
- `load_mall_customers_data()`: Cached data loading with error handling
- `preprocess_data()`: Feature scaling and preprocessing
- `load_and_preprocess_data()`: Combined loading and preprocessing

#### Application Comparison

| Application | Primary Focus | Algorithm | Complexity | Best For |
|-------------|---------------|-----------|------------|----------|
| **app3.py** | Enhanced hybrid clustering | Hybrid + Custom graphs | High | Production use, comprehensive analysis |
| **app2.py** | Research implementation | EM-style weighted fusion | Very High | Research, algorithm development |
| **main1.py** | Customer segmentation | Hybrid with preprocessing | Medium | Customer analytics |
| **main2.py** | Wine quality analysis | Hybrid with PCA | Medium | Wine industry analysis |
| **app.py** | Basic hybrid | Simple fusion | Low | Learning, prototyping |

## 📚 API Reference

### HybridClustering Class

```python
class HybridClustering:
    """
    Hybrid clustering algorithm combining K-means and Spectral clustering.
    
    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters to form and centroids to generate.
        
    alpha : float, default=0.5
        Fusion weight parameter controlling the balance between
        centroid-based and graph-based clustering.
        - 0.0: Pure spectral clustering
        - 1.0: Pure K-means clustering
        - 0.5: Balanced hybrid approach
    
    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Labels of each point
        
    centroids_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers
        
    spectral_labels_ : ndarray of shape (n_samples,)
        Pure spectral clustering labels for comparison
    """
    
    def __init__(self, n_clusters=3, alpha=0.5):
        """Initialize hybrid clustering with specified parameters."""
        
    def create_similarity_graph(self, X):
        """
        Create similarity matrix using RBF (Gaussian) kernel.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data
            
        Returns
        -------
        similarities : ndarray of shape (n_samples, n_samples)
            Pairwise similarity matrix
        """
        
    def fit(self, X):
        """
        Compute hybrid clustering on input data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to cluster
            
        Returns
        -------
        self : object
            Returns the instance itself
        """
```

### Utility Functions

#### Data Loading and Preprocessing
```python
@st.cache_data
def load_mall_customers_data():
    """
    Load Mall Customers dataset with caching and error handling.
    
    Returns
    -------
    df : pandas.DataFrame or None
        Customer dataset with columns: CustomerID, Gender, Age, 
        Annual Income (k$), Spending Score (1-100)
    """

def preprocess_data(df):
    """
    Preprocess customer data for clustering.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Raw customer dataset
        
    Returns
    -------
    X_scaled : ndarray
        Standardized feature matrix
    features : list
        List of feature column names
    """
```

#### Visualization Functions
```python
def plot_clusters(X_scaled, labels, features, centroids=None, title="Clustering Results"):
    """
    Create scatter plot of clustering results.
    
    Parameters
    ----------
    X_scaled : ndarray of shape (n_samples, n_features)
        Scaled input data
    labels : ndarray of shape (n_samples,)
        Cluster labels
    features : list
        Feature names for axis labels
    centroids : ndarray, optional
        Cluster centroids to plot
    title : str, default="Clustering Results"
        Plot title
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated plot figure
    """

def plot_network(X_scaled, labels, similarities, threshold=90, title="Network Graph"):
    """
    Create network graph visualization of clustering results.
    
    Parameters
    ----------
    X_scaled : ndarray
        Scaled input data
    labels : ndarray
        Cluster labels
    similarities : ndarray
        Pairwise similarity matrix
    threshold : float, default=90
        Percentile threshold for edge inclusion
    title : str
        Graph title
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Network graph figure
    """
```

### Advanced API (app2.py)

```python
def hybrid_clustering(X, n_clusters, alpha=0.5, max_iter=100, tol=1e-4):
    """
    Advanced hybrid clustering with EM-style optimization.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data
    n_clusters : int
        Number of clusters
    alpha : float, default=0.5
        Weight parameter for fusion
    max_iter : int, default=100
        Maximum number of iterations
    tol : float, default=1e-4
        Convergence tolerance
        
    Returns
    -------
    cluster_assignments : ndarray
        Final cluster labels
    centroids : ndarray
        Final cluster centroids
    """

def compute_dissimilarities(X):
    """
    Compute pairwise squared Euclidean distances.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data
        
    Returns
    -------
    W : ndarray of shape (n_samples, n_samples)
        Pairwise dissimilarity matrix
    """
```

## 🔧 Troubleshooting

### Common Installation Issues

#### 1. Package Installation Failures
```bash
# Clear pip cache and reinstall
pip cache purge
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

#### 2. Virtual Environment Issues
```bash
# Remove and recreate virtual environment
rmdir /s clustering_env  # Windows
rm -rf clustering_env    # macOS/Linux

python -m venv clustering_env
clustering_env\Scripts\activate  # Windows
pip install -r requirements.txt
```

#### 3. Streamlit Port Conflicts
```bash
# Use alternative port
streamlit run app3.py --server.port 8502

# Kill existing Streamlit processes (Windows)
taskkill /f /im streamlit.exe

# Kill existing Streamlit processes (macOS/Linux)
pkill -f streamlit
```

### Data-Related Issues

#### 1. File Not Found Errors
**Problem**: `FileNotFoundError: Mall_Customers.csv not found`

**Solutions**:
```bash
# Verify file location
dir Mall_Customers.csv  # Windows
ls -la Mall_Customers.csv  # macOS/Linux

# Check current directory
pwd
```

**Alternative**: Modify file path in code:
```python
# In load_mall_customers_data() function
df = pd.read_csv('data/Mall_Customers.csv')  # If in subdirectory
df = pd.read_csv('C:/full/path/to/Mall_Customers.csv')  # Absolute path
```

#### 2. Data Format Issues
**Problem**: CSV parsing errors or unexpected columns

**Solution**: Verify data format:
```python
import pandas as pd
df = pd.read_csv('Mall_Customers.csv')
print(df.head())
print(df.columns.tolist())
print(df.dtypes)
```

### Performance Issues

#### 1. Memory Errors
**Symptoms**: Out of memory errors with large datasets

**Solutions**:
- Reduce dataset size for testing
- Use data sampling:
```python
# Sample 10% of data
df_sample = df.sample(frac=0.1, random_state=42)
```
- Close other applications
- Increase virtual memory

#### 2. Slow Clustering Performance
**Optimizations**:
```python
# Reduce similarity threshold
plot_network(X_scaled, labels, similarities, threshold=95)  # Higher threshold

# Limit iterations for app2.py
hybrid_clustering(X, n_clusters=5, max_iter=50)  # Fewer iterations

# Use PCA for dimensionality reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)
```

### Visualization Issues

#### 1. Matplotlib Backend Errors
```bash
# Set backend explicitly
export MPLBACKEND=Agg  # Linux/macOS
set MPLBACKEND=Agg     # Windows

# Or in Python code:
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```

#### 2. NetworkX Layout Issues
```python
# Alternative layout algorithms
pos = nx.spring_layout(G, k=0.5, iterations=20)  # Adjust parameters
pos = nx.kamada_kawai_layout(G)  # Alternative layout
pos = nx.circular_layout(G)      # Simple circular layout
```

### Algorithm-Specific Issues

#### 1. Poor Clustering Results
**Diagnostics**:
```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Evaluate clustering quality
sil_score = silhouette_score(X_scaled, labels)
ch_score = calinski_harabasz_score(X_scaled, labels)
print(f"Silhouette Score: {sil_score:.3f}")
print(f"Calinski-Harabasz Score: {ch_score:.3f}")
```

**Improvements**:
- Adjust number of clusters based on silhouette score
- Try different alpha values
- Ensure proper data preprocessing (scaling)
- Check for outliers in data

#### 2. Convergence Issues (app2.py)
```python
# Increase tolerance or iterations
cluster_assignments, centroids = hybrid_clustering(
    X, n_clusters=5, alpha=0.5, max_iter=200, tol=1e-6
)

# Add convergence monitoring
if iteration == max_iter - 1:
    print("Warning: Maximum iterations reached without convergence")
```

### Development Environment Issues

#### 1. IDE Integration Problems
**VS Code**: Install Python extension and select correct interpreter
```bash
# Check Python interpreter
which python  # macOS/Linux
where python  # Windows
```

**Jupyter**: Install Jupyter extension for notebook support
```bash
pip install jupyter
```

#### 2. Git Integration
```bash
# Initialize repository if needed
git init
git add .
git commit -m "Initial commit"

# Handle line ending issues
git config core.autocrlf true    # Windows
git config core.autocrlf input   # macOS/Linux
```

## 🤝 Contribution Guidelines

### Getting Started

#### 1. Fork and Clone
```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/yourusername/fusion-of-centroid-based-clustering-with-graph-clustering.git
cd "fusion of centroid-based clustering with graph clustering"
```

#### 2. Create Development Environment
```bash
# Create and activate virtual environment
python -m venv dev_env
dev_env\Scripts\activate  # Windows
source dev_env/bin/activate  # macOS/Linux

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 jupyter  # Additional dev tools
```

#### 3. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/your-bugfix-name
```

### Code Standards

#### 1. Python Style Guidelines
Follow PEP 8 with these specific requirements:
- Maximum line length: 88 characters (Black formatter default)
- Use double quotes for strings
- Function and variable names: `snake_case`
- Class names: `PascalCase`
- Constants: `UPPER_CASE`

#### 2. Code Formatting
```bash
# Auto-format code with Black
black *.py

# Check code style
flake8 *.py --max-line-length=88 --ignore=E203,W503
```

#### 3. Documentation Standards
```python
def example_function(param1, param2):
    """
    Brief description of the function.
    
    Detailed description if needed. Explain the algorithm,
    mathematical foundations, or complex logic.
    
    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type, optional
        Description of param2 (default: value)
        
    Returns
    -------
    result : type
        Description of return value
        
    Raises
    ------
    ValueError
        When parameter validation fails
        
    Examples
    --------
    >>> result = example_function(arg1, arg2)
    >>> print(result)
    Expected output
    
    Notes
    -----
    Additional notes, mathematical formulas, or references.
    """
    pass
```

### Testing Requirements

#### 1. Unit Tests
Create tests for new functions:
```python
# test_clustering.py
import pytest
import numpy as np
from app3 import HybridClustering

def test_hybrid_clustering_initialization():
    """Test HybridClustering initialization with different parameters."""
    clustering = HybridClustering(n_clusters=5, alpha=0.3)
    assert clustering.n_clusters == 5
    assert clustering.alpha == 0.3

def test_hybrid_clustering_fit():
    """Test clustering fit method with sample data."""
    # Generate sample data
    np.random.seed(42)
    X = np.random.rand(100, 2)
    
    clustering = HybridClustering(n_clusters=3, alpha=0.5)
    clustering.fit(X)
    
    assert hasattr(clustering, 'labels_')
    assert len(clustering.labels_) == 100
    assert len(np.unique(clustering.labels_)) <= 3

def test_similarity_graph_creation():
    """Test similarity graph creation."""
    clustering = HybridClustering()
    X = np.array([[0, 0], [1, 1], [2, 2]])
    similarities = clustering.create_similarity_graph(X)
    
    assert similarities.shape == (3, 3)
    assert np.allclose(np.diag(similarities), 1.0)  # Diagonal should be 1
```

#### 2. Integration Tests
```python
def test_streamlit_app_loading():
    """Test that Streamlit apps can be imported without errors."""
    try:
        import app3
        import main1
        import main2
        assert True
    except ImportError as e:
        pytest.fail(f"Import error: {e}")

def test_data_loading():
    """Test data loading functions."""
    from app3 import load_mall_customers_data
    
    df = load_mall_customers_data()
    if df is not None:
        assert 'CustomerID' in df.columns
        assert 'Annual Income (k$)' in df.columns
        assert 'Spending Score (1-100)' in df.columns
```

#### 3. Running Tests
```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
pytest test_clustering.py -v

# Run with coverage
pip install pytest-cov
pytest test_clustering.py --cov=app3 --cov-report=html
```

### Contribution Types

#### 1. Bug Fixes
- **Scope**: Fix existing functionality without breaking changes
- **Requirements**: Include test case that reproduces the bug
- **Documentation**: Update relevant docstrings and comments

#### 2. New Features
- **Scope**: Add new clustering algorithms, visualization types, or datasets
- **Requirements**: 
  - Comprehensive tests for new functionality
  - Documentation updates
  - Example usage in docstrings
- **Examples**:
  - New clustering algorithm implementations
  - Additional visualization options
  - New dataset support
  - Performance optimizations

#### 3. Documentation Improvements
- **Scope**: README updates, code comments, docstring improvements
- **Examples**:
  - Tutorial additions
  - API documentation enhancements
  - Example code snippets
  - Troubleshooting guides

#### 4. Performance Optimizations
- **Requirements**: Benchmark comparisons showing improvement
- **Documentation**: Performance testing methodology and results

### Submission Process

#### 1. Pre-submission Checklist
- [ ] Code follows style guidelines (Black formatting, PEP 8)
- [ ] All tests pass (`pytest test_clustering.py`)
- [ ] New features include tests and documentation
- [ ] No breaking changes to existing API
- [ ] Commit messages are clear and descriptive

#### 2. Commit Message Format
```bash
# Format: <type>(<scope>): <description>
# Examples:
git commit -m "feat(clustering): add DBSCAN hybrid implementation"
git commit -m "fix(visualization): resolve NetworkX layout error"
git commit -m "docs(readme): add troubleshooting section"
git commit -m "test(app3): add unit tests for HybridClustering"
```

#### 3. Pull Request Guidelines
1. **Title**: Clear, descriptive summary of changes
2. **Description**: 
   - What changes were made and why
   - How to test the changes
   - Any breaking changes or migration notes
   - Screenshots for UI changes
3. **Review**: Address all reviewer comments
4. **Testing**: Ensure CI/CD checks pass


## Changes Made
- [ ] Added new feature X
- [ ] Fixed bug Y
- [ ] Updated documentation Z

## Testing
- [ ] All existing tests pass
- [ ] Added new tests for new functionality
- [ ] Manually tested with sample data

## Screenshots (if applicable)
[Include screenshots for UI changes]

## Breaking Changes
None / [Describe any breaking changes]
```

### Code Review Process

#### 1. Review Criteria
- **Functionality**: Does the code work as intended?
- **Quality**: Is the code readable, maintainable, and efficient?
- **Testing**: Are there adequate tests?
- **Documentation**: Is the code well-documented?
- **Style**: Does it follow project conventions?

#### 2. Review Timeline
- Initial review within 2-3 business days
- Follow-up reviews within 1 business day
- Merge after approval from at least one maintainer

### Recognition

Contributors will be recognized in:
- GitHub contributors list
- CHANGELOG.md for significant contributions
- Special recognition for major features or improvements

## 📄 License Information

This project is released under the **MIT License**.

### MIT License Summary
- ✅ **Commercial use**: Use for commercial purposes
- ✅ **Modification**: Modify and adapt the code
- ✅ **Distribution**: Distribute original or modified versions
- ✅ **Private use**: Use privately without restrictions
- ❗ **Limitation**: No liability or warranty provided
- ❗ **License and copyright notice**: Must include license in distributions

### Full License Text
```
MIT License

Copyright (c) 2025 Fusion Clustering Project Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 📞 Contact/Support Details

### Getting Help

#### 1. Documentation and Self-Help
- **README**: This comprehensive guide
- **Code Comments**: Detailed inline documentation
- **Docstrings**: Function and class documentation
- **Examples**: Usage examples throughout the codebase

#### 2. Community Support
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/fusion-of-centroid-based-clustering-with-graph-clustering/issues) for:
  - Bug reports
  - Feature requests
  - Usage questions
  - Documentation improvements

#### 3. Issue Reporting Guidelines

**Bug Reports** - Include:
```markdown
**Environment Information:**
- OS: [Windows 10/macOS 11/Ubuntu 20.04]
- Python version: [3.8.5]
- Package versions: `pip list | grep -E "(streamlit|pandas|sklearn|networkx)"`


**Feature Requests** - Include:
- Clear description of the proposed feature
- Use case and motivation
- Possible implementation approach
- Alternative solutions considered

#### 4. Response Times
- **Bug reports**: 1-2 business days for initial response
- **Feature requests**: 3-5 business days for evaluation
- **Questions**: 1-3 business days depending on complexity

### Development Team

**Primary Maintainer**: [Ashrith Velisoju]
- GitHub: [@ashrithvelisoju](https://github.com/ashrithvelisoju)
- Email: ashrithvelisoju@gmail.com (for security issues only)

**Contributing Developers**: See [Contributors](https://github.com/ashrithvelisoju/fusion-of-centroid-based-clustering-with-graph-clustering/contributors)

### Security Issues

For security-related issues, please email directly instead of creating public issues:
- Email: ashrithvelisoju@gmail.com
- PGP Key: [Link to public key if applicable]

### Academic and Research Inquiries

This project implements hybrid clustering techniques that may be relevant for academic research:
- **Methodology**: Fusion of centroid-based and graph-based clustering
- **Applications**: Customer segmentation, data analysis, machine learning
- **Citations**: If you use this work in academic research, please cite appropriately

### Commercial Support

For commercial implementations or consulting:
- Custom algorithm development
- Large-scale deployment assistance
- Training and workshops
- Contact: business@yourproject.com

---

## 🏆 Acknowledgments

### Research Foundations
This implementation builds upon established research in:
- **K-means Clustering**: Lloyd's algorithm and variants
- **Spectral Clustering**: Graph Laplacian-based methods
- **Hybrid Clustering**: Fusion methodologies for improved performance

### Open Source Libraries
Special thanks to the maintainers of:
- **Scikit-learn**: Machine learning algorithms and utilities
- **NetworkX**: Graph analysis and visualization
- **Streamlit**: Interactive web application framework
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Matplotlib & Seaborn**: Visualization libraries

### Contributors
- [Ashrith Velisoju] - Initial implementation and documentation

----

**Version**: 1.0.0  
**Last Updated**: June 21, 2025  
**Compatibility**: Python 3.7+ | All major operating systems
