import numpy as np
from data.data import load_data_multi
from image_utils import *

def compute_feature_stats(data): 
    """
    Computes the mean and variance for each feature
    in the dataset.
    
    Args:
        data (ndarray): (m, n) Input feature matrix
    
    Returns:
        means (ndarray): (n,) Mean of each feature
        variances (ndarray): (n,) Variance of each feature
    """
    num_samples = data.shape[0]
    means = np.sum(data, axis=0) / num_samples
    variances = np.sum((data - means)**2, axis=0) / num_samples
    return means, variances

def find_optimal_threshold(true_labels, predicted_probs): 
    """
    Determines the optimal threshold (epsilon) to identify outliers
    based on validation set probabilities and ground truth labels.
    
    Args:
        true_labels (ndarray): Ground truth binary labels (0 or 1)
        predicted_probs (ndarray): Probability estimates for validation set
        
    Returns:
        best_threshold (float): Chosen threshold value
        best_f1 (float): F1 score corresponding to the best threshold
    """ 

    best_threshold = 0
    best_f1 = 0
    
    step = (max(predicted_probs) - min(predicted_probs)) / 1000
    
    for threshold in np.arange(min(predicted_probs), max(predicted_probs), step):
        # Classify points as anomalies if their probability is below the threshold
        is_anomaly = (predicted_probs < threshold)

        true_positive = np.sum((is_anomaly == 1) & (true_labels == 1))
        false_positive = np.sum((is_anomaly == 1) & (true_labels == 0))
        false_negative = np.sum((is_anomaly == 0) & (true_labels == 1))

        precision = true_positive / (true_positive + false_positive + 1e-10)
        recall = true_positive / (true_positive + false_negative + 1e-10)
        
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)

        if f1_score > best_f1:
            best_f1 = f1_score
            best_threshold = threshold
        
    return best_threshold, best_f1

def multivariate_gaussian(X, mean, var):
    """
    Computes the probability density of the multivariate Gaussian distribution
    assuming independent features (diagonal covariance matrix).
    
    Args:
        X (ndarray): (m, n) Data matrix where m is the number of samples and n is the number of features
        mean (ndarray): (n,) Mean vector of the distribution
        var (ndarray): (n,) Variance vector (diagonal of covariance matrix)
    
    Returns:
        probs (ndarray): (m,) Probability density for each sample
    """
    n = X.shape[1]
    eps = 1e-9  # for numerical stability in case of zero variance

    # Compute the probability density for each example
    coeff = 1 / np.sqrt((2 * np.pi) ** n * np.prod(var + eps))
    exponent = -0.5 * np.sum(((X - mean) ** 2) / (var + eps), axis=1)
    probs = coeff * np.exp(exponent)
    
    return probs


if __name__ == "__main__":
    X_train, X_val, y_val = load_data_multi()
    # image, X_img = load_and_prepare_image('data/images/k-means-test.jpg')

    # X_train, X_val, _, y_val = get_sumilated_lables_from_image(X_img)

    mu, var = compute_feature_stats(X_train)
    p_train = multivariate_gaussian(X_train, mu, var)
    p_val = multivariate_gaussian(X_val, mu, var)

    epsilon, F1 = find_optimal_threshold(y_val, p_val)

    print("Best epsilon:", epsilon)
    print("Best F1 score:", F1)
    print("Detected anomalies in training set:", np.sum(p_train < epsilon))

    # Visualize anomalies on the image
    # anomaly_mask = multivariate_gaussian(X_img, mu, var) < epsilon
    # visualize_detected_image_anomalies(image, anomaly_mask)