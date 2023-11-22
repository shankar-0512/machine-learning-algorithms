import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# set the random seed to ensure reproducibility of results
np.random.seed(1976)

##################### SILHOUETTE COEFFICIENT #####################
def silhouette_coefficient(dataset, cluster_assignments, centroids):
    
    # Get the number of clusters in the clustering solution
    num_clusters = len(np.unique(cluster_assignments))
    
    # Initialize an array to store the silhouette score for each data point
    silhouette_scores = np.zeros(dataset.shape[0])
    
    # Loop over each data point
    for i in range(dataset.shape[0]):
        
        # Compute the mean distance to all other points in the same cluster
        cluster_i = cluster_assignments == cluster_assignments[i]
        distances = np.linalg.norm(dataset[cluster_i] - dataset[i], axis=1)
        a = np.mean(distances)
        
        # Compute the mean distance to all points in the nearest other cluster
        b = np.inf
        
        for j in range(num_clusters):
            
            if j != cluster_assignments[i] and np.sum(cluster_assignments == j) > 0:
                
                cluster_i = cluster_assignments == j
                distances = np.linalg.norm(dataset[cluster_i] - dataset[i], axis=1)
                b = min(b, np.mean(distances))
                
        # Compute the silhouette score for this data point
        if np.isnan(a) or np.isnan(b) or np.isinf(a) or np.isinf(b):
            silhouette_scores[i] = 0
        else:
            silhouette_scores[i] = (b - a) / max(a, b)
            
    # Compute the overall silhouette score by taking the mean of the scores for each data point
    silhouette_coefficient = np.mean(silhouette_scores)
    return silhouette_coefficient


class KMeans:
    
    def __init__(self, max_iterations):
        self.max_iterations = max_iterations
        
    def cluster(self, dataset, k):
        
        # Initialize centroids randomly
        centroids = dataset[np.random.choice(dataset.shape[0], k, replace=False), :]
        
        # Store the current centroids for the next iteration
        old_centroids = centroids.copy()
        
        for iteration in range(self.max_iterations):
            
            # Compute the distances between each data point and each centroid
            distances = np.linalg.norm(dataset[:, np.newaxis, :] - centroids, axis=2)
            
            # Assign each data point to the closest centroid
            cluster_assignments = np.argmin(distances, axis=1)
            
            # Update centroids by computing the mean of all the data points assigned to each cluster
            for j in range(k):
                cluster_points = dataset[cluster_assignments == j]
                if cluster_points.shape[0] > 0:
                    centroids[j] = np.mean(cluster_points, axis=0)
            
            # Check for convergence
            if np.allclose(old_centroids, centroids):
                break
        
        return cluster_assignments, centroids

    
class KMeansPlusPlus:
    
    def __init__(self, max_iterations):
        
        # Initializing class attributes
        self.max_iterations = max_iterations
        self.silhouette_coefficient = silhouette_coefficient
    
    def cluster(self, dataset, k):
        
        # Select the first center uniformly at random
        centroids = [dataset[np.random.choice(dataset.shape[0])]]
        
        # Select the remaining k-1 centers using k-means++ algorithm
        for i in range(1, k):
            
            # Calculate distances to the nearest center for each point
            
            # Define an empty array to hold the minimum distances
            distances = np.empty(len(dataset))
            
            # Loop over each data point
            for i, x in enumerate(dataset):
                
                # Calculate the distance between the data point and each centroid
                centroidsDist = [np.linalg.norm(x - c)**2 for c in centroids]
                
                # Find the minimum distance among all centroids
                minDist = min(centroidsDist)
                
                # Add the minimum distance to the distances array
                distances[i] = minDist
            
            # Calculate probability weights for each point
            weights = distances / distances.sum()
            
            # Select next center randomly based on probability weights
            centroids.append(dataset[np.random.choice(dataset.shape[0], p=weights)])
            
        # Store the current centroids for the next iteration
        old_centroids = centroids.copy()
        
        # NORMAL K-MEANS IMPLEMENTATION
        
        for iteration in range(self.max_iterations):
            
            # Assign data points to clusters
            distances = np.linalg.norm(dataset[:, np.newaxis, :] - centroids, axis=2)
            cluster_assignments = np.argmin(distances, axis=1)
            
            # Update centroids
            for j in range(k):
                cluster_points = dataset[cluster_assignments == j]
                if cluster_points.shape[0] > 0:
                    centroids[j] = np.mean(cluster_points, axis=0)
                    
            # Check for convergence
            if np.allclose(old_centroids, centroids):
                break
        
        # Compute the silhouette coefficient for the resulting clustering
        silhouette_coefficient = self.silhouette_coefficient(dataset, cluster_assignments, centroids)

        return silhouette_coefficient
                                                             
                                                             
class BisectingKMeans:
    
    def __init__(self, max_iterations):
        
        # Initializing class attributes
        self.max_iterations = max_iterations
        self.silhouette_coefficient = silhouette_coefficient
    
    def cluster(self, dataset):
        
        # Initialize with all data points in one cluster
        # Create an array to hold cluster assignments, initialized to 0
        cluster_assignments = np.zeros(dataset.shape[0], dtype=int)
        
        # Compute the initial centroid as the mean of all data points
        centroids = np.mean(dataset, axis=0, keepdims=True)
        
        # Create an array to hold silhouette coefficients for each iteration of the bisecting k-means algorithm
        silhouette_coefficient = np.zeros(8)

        # Perform bisecting k-means until the desired number of clusters is reached
        for i in range(1, 9):
            
            # Find the cluster with the largest sum of squared distances to its centroid
            max_cluster = np.argmax([np.sum((dataset[cluster_assignments == i] - centroids[i])**2) 
                                     for i in range(centroids.shape[0])])


            # Split the largest cluster into two
            # Find the indices of data points in the largest cluster
            split_indices = np.where(cluster_assignments == max_cluster)[0]
            
            # Extract the data points from the largest cluster
            split_data = dataset[split_indices]
            
            # Create a KMeans object with a maximum of 100 iterations
            clustering = KMeans(max_iterations = 100)
            
            # Cluster the split data using KMeans with k=2 and get the assignments and centroids
            split_assignments, split_centroids = clustering.cluster(split_data, 2)

            # Update the cluster assignments and centroids
            # Create a new array to hold updated cluster assignments, initialized to 0
            new_assignments = np.zeros(dataset.shape[0], dtype=int) 
            
            # Assign data points in the first split cluster to the largest cluster
            new_assignments[split_indices[split_assignments == 0]] = max_cluster 
            
            # Assign data points in the second split cluster to a new cluster
            new_assignments[split_indices[split_assignments == 1]] = centroids.shape[0]
            
            # Update the cluster assignments for the largest cluster
            cluster_assignments[cluster_assignments == max_cluster] = new_assignments[cluster_assignments == max_cluster] 
            
            # Update the centroid for the largest cluster with the centroid of the first split cluster
            centroids[max_cluster] = split_centroids[0] 
            
            # Add the centroid of the second split cluster as a new centroid
            centroids = np.vstack([centroids, split_centroids[1]])

            # Compute the silhouette coefficient for the current clustering and store it in the array
            silhouette_coefficient[i-1] = self.silhouette_coefficient(dataset, cluster_assignments, centroids)

        return silhouette_coefficient


##################### DATASET MANIPULAION #####################
dataset = pd.read_csv("dataset", delimiter=" ", header=None)
dataset = (np.array(dataset)[:, 1:]).astype(np.float64)

kMeansSilhouette = np.zeros(8)
kMeansPlusPlusSilhouette = np.zeros(8)

##################### K-MEANS CLUSTERING #####################
for k in range(2, 10):
    
    clustering = KMeans(max_iterations = 100)
    cluster_assignments, centroids = clustering.cluster(dataset, k)
    kMeansSilhouette[k-2] = silhouette_coefficient(dataset, cluster_assignments, centroids)
    
##################### K-MEANS++ CLUSTERING #####################
for k in range(2, 10):
    
    clustering = KMeansPlusPlus(max_iterations = 100)
    kMeansPlusPlusSilhouette[k-2] = clustering.cluster(dataset, k)

##################### BISECTING K-MEANS CLUSTERING #####################    
clustering = BisectingKMeans(max_iterations = 100)
bisectingKMeansSilhouette = clustering.cluster(dataset)


##################### PLOTS ##################### 
# define the figure with one subplot
fig, ax = plt.subplots(1, 3, figsize=(16, 4))

# plot silhouette scores
ax[0].set_title("K-means", fontsize=14, weight = "bold")
ax[0].set_xlabel("k", fontsize=14)
ax[0].set_ylabel("Silhouette Coefficient", fontsize=14)
ax[0].tick_params(axis="both", labelsize=12)
ax[0].grid(linestyle='--', alpha=0.7)
ax[0].plot(range(2, 10), kMeansSilhouette, lw=2, color='#0017FF', marker="o", ms=5, mfc="black", mec="black")

# add background color
fig.patch.set_facecolor('#F5F5F5')
ax[0].set_facecolor('#FFFFFF')

# plot silhouette scores
ax[1].set_title("K-means ++", fontsize=14, weight = "bold")
ax[1].set_xlabel("k", fontsize=14)
ax[1].set_ylabel("Silhouette Coefficient", fontsize=14)
ax[1].tick_params(axis="both", labelsize=12)
ax[1].grid(linestyle='--', alpha=0.7)
ax[1].plot(range(2, 10), kMeansPlusPlusSilhouette, lw=2, color='#FF0000', marker="o", ms=5, mfc="black", mec="black")

# add background color
fig.patch.set_facecolor('#F5F5F5')
ax[1].set_facecolor('#FFFFFF')

# plot silhouette scores
ax[2].set_title("Bisecting k-means", fontsize=14, weight = "bold")
ax[2].set_xlabel("s", fontsize=14)
ax[2].set_ylabel("Silhouette Coefficient", fontsize=14)
ax[2].tick_params(axis="both", labelsize=12)
ax[2].grid(linestyle='--', alpha=0.7)
ax[2].plot(range(2, 10), bisectingKMeansSilhouette, lw=2, color='#08FF00', marker="o", ms=5, mfc="black", mec="black")

# add background color
fig.patch.set_facecolor('#F5F5F5')
ax[2].set_facecolor('#FFFFFF')

plt.subplots_adjust(wspace = 0.4)

plt.show()