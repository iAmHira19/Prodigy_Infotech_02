import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv('Mall_Customers.csv')

# Explore the data
print(data.head())

# Select relevant features
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Data preprocessing (if needed)
# Example: Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters (k) using the elbow method or other techniques

# Initialize K-means
kmeans = KMeans(n_clusters=5, random_state=42)

# Fit K-means
kmeans.fit(X_scaled)

# Predict the clusters
clusters = kmeans.predict(X_scaled)

# Visualize the clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.8, edgecolors='k')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.title('K-means Clustering of Customers')
plt.colorbar()
plt.show()
