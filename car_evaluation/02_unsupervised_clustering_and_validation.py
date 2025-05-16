print("                                                                     ")
print("                                                                     ")
print("                 .2  -- Unsupervised learning techniques  --")
print("                                                                     ")

import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

# Load the encoded dataset
df = pd.read_csv('car_evaluation_encoded.csv')

# Separate features and true labels (which we will NOT use for clustering but will use for ARI validation)
X = df.drop('class', axis=1)
y_true = df['class']

print("Applying KMeans clustering...")
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

print("Applying Agglomerative Clustering...")
agglo = AgglomerativeClustering(n_clusters=4)
agglo_labels = agglo.fit_predict(X)

# Cluster validation metrics
print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                           KMeans Validation ")
print("---------------------------------------------------------------------------------")
print("Silhouette Score:", silhouette_score(X, kmeans_labels))
print("Davies-Bouldin Score:", davies_bouldin_score(X, kmeans_labels))
print("Adjusted Rand Index vs. true labels:", adjusted_rand_score(y_true, kmeans_labels))


print("                                                                     ")
print("---------------------------------------------------------------------------------")
print("                          Agglomerative Clustering Validation ")
print("---------------------------------------------------------------------------------")
print("Silhouette Score:", silhouette_score(X, agglo_labels))
print("Davies-Bouldin Score:", davies_bouldin_score(X, agglo_labels))
print("Adjusted Rand Index vs. true labels:", adjusted_rand_score(y_true, agglo_labels))
