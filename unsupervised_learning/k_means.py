import numpy as np
import matplotlib.pyplot as plt

def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example using vectorized NumPy operations.
    
    Args:
        X (ndarray): (m, n) Input values      
        centroids (ndarray): (K, n) centroids
    
    Returns:
        idx (ndarray): (m,) Indices of the closest centroids
    """
    distances = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    return np.argmin(distances, axis=1)


def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the 
    data points assigned to each centroid.
    
    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each 
                       example in X. Concretely, idx[i] contains the index of 
                       the centroid closest to example i
        K (int):       number of centroids
    
    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """
    n = X.shape[1]
    centroids = np.zeros((K, n))
    
    for i in range(K):
        points = X[idx == i]
        if len(points) > 0:
            centroids[i] = np.mean(points, axis=0)
        else:
            centroids[i] = np.zeros(n)  # np.random.rand(n)
    return centroids

def kMeans_init_centroids(X, K):
    """
    This function initializes K centroids that are to be 
    used in K-Means on the dataset X
    
    Args:
        X (ndarray): Data points 
        K (int):     number of centroids/clusters
    
    Returns:
        centroids (ndarray): Initialized centroids
    """
    
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[:K]]
    
    return centroids

def run_kMeans(X, initial_centroids, max_iters=10):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    idx = np.zeros(X.shape[0])
    for i in range(max_iters):
        print("K-Means iteration %d/%d" % (i + 1, max_iters))
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)
    return centroids, idx

def run_kMeans_multiple_inits(X, K, max_iters=10, n_init=10):
    best_centroids = None
    best_idx = None
    best_inertia = float('inf')

    for i in range(n_init):
        print(f"Initialization {i+1}/{n_init}")
        initial_centroids = kMeans_init_centroids(X, K)
        centroids, idx = run_kMeans(X, initial_centroids, max_iters)
        inertia = np.sum((X - centroids[idx])**2)

        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids
            best_idx = idx

    return best_centroids, best_idx

if __name__ == "__main__":
    original_img = plt.imread('data/images/k-means-test.jpg')
    X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))
    K = 5
    max_iters = 10

    # initial_centroids = kMeans_init_centroids(X_img, K)
    # centroids, idx = run_kMeans(X_img, initial_centroids, max_iters)
    
    centroids, idx = run_kMeans_multiple_inits(X_img, K, max_iters, n_init=3)
    # Run test
    idx = find_closest_centroids(X_img, centroids)
    X_recovered = centroids[idx, :] 
    X_recovered = np.reshape(X_recovered, original_img.shape)
    if original_img.dtype == np.uint8:
        X_recovered = np.clip(X_recovered, 0, 255).astype(np.uint8)
    else:
        X_recovered = np.clip(X_recovered, 0, 1)

    fig, ax = plt.subplots(1,2, figsize=(16,16))
    plt.axis('off')

    ax[0].imshow(original_img)
    ax[0].set_title('Original')
    ax[0].set_axis_off()

    ax[1].imshow(X_recovered)
    ax[1].set_title('Compressed with %d colours'%K)
    ax[1].set_axis_off()
    plt.show()