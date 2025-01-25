import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
customer = pd.read_csv("N:\\zeotap\\customer.csv.csv")
transaction = pd.read_csv("N:\\zeotap\\transaction.csv.csv")

# Merging customer and transaction data based on CustomerID
customer_transaction = pd.merge(transaction, customer, on='CustomerID', how='inner')

# Feature engineering: Use 'TotalValue' and 'Quantity' as features for clustering
customer_agg = customer_transaction.groupby('CustomerID').agg({
    'TotalValue': 'sum',  # Total spending
    'Quantity': 'sum',    # Total number of items purchased
}).reset_index()

# Merge with customer profile information (e.g., Region, SignupDate)
customer_data = pd.merge(customer, customer_agg, on='CustomerID', how='inner')

# Drop any columns that won't be used for clustering (e.g., CustomerID, CustomerName, SignupDate)
features = customer_data[['TotalValue', 'Quantity']]

# Normalize the features for better clustering performance
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Determine the optimal number of clusters using the elbow method
inertia = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_scaled)
    inertia.append(kmeans.inertia_)

# Plot inertia to find the "elbow"
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Based on the elbow plot, choose the optimal number of clusters (let's assume k=4 here)
k = 4

# Apply KMeans clustering with k=4
kmeans = KMeans(n_clusters=k, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(features_scaled)

# Clustering Metrics
silhouette_avg = silhouette_score(features_scaled, customer_data['Cluster'])
print(f"Silhouette Score: {silhouette_avg}")

# DB Index (Dunn's Index)
def db_index(data, labels, centroids):
    dist = cdist(data, centroids, 'euclidean')
    min_dist = np.min(dist, axis=1)
    max_dist = np.max(dist, axis=1)
    db_index_value = np.mean(max_dist / min_dist)
    return db_index_value

centroids = kmeans.cluster_centers_
db_index_value = db_index(features_scaled, customer_data['Cluster'], centroids)
print(f"Dunn's Index (DB Index): {db_index_value}")

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=customer_data['TotalValue'], y=customer_data['Quantity'], hue=customer_data['Cluster'], palette='Set1', s=100, marker='o')
plt.title('Customer Segmentation (K-Means Clustering)')
plt.xlabel('Total Value (Spending)')
plt.ylabel('Quantity (Number of Items Purchased)')
plt.legend(title='Cluster')
plt.show()

# Save the results
lookalike_map = customer_data[['CustomerID', 'Cluster']]
lookalike_map.columns = ['CustomerID', 'Cluster']
lookalike_map.to_csv("N:\\zeotap\\lookalike_clusters.csv", index=False)

print("Clustering results saved to 'lookalike_clusters.csv'")
