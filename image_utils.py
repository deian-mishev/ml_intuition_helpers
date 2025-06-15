import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_and_prepare_image(path, normalize=True):
    """
    Loads an image and reshapes it into a 2D array of RGB values.

    Args:
        path (str): Path to the image file
        normalize (bool): Whether to normalize pixel values to [0, 1]

    Returns:
        image (ndarray): Original image (H, W, 3)
        X_img (ndarray): Reshaped image data (H*W, 3)
    """
    image = plt.imread(path)
    if normalize and image.dtype == np.uint8:
        image = image / 255.0
    X_img = np.reshape(image, (-1, 3))
    return image, X_img

def get_sumilated_lables_from_image(X, anomaly_fraction=0.01, seed=42):
    """
    Simulates validation labels with a given fraction of anomalies.

    Args:
        X (ndarray): Data matrix
        anomaly_fraction (float): Fraction of points to label as anomalies
        seed (int): Random seed for reproducibility

    Returns:
        X_train (ndarray): Training data
        X_val (ndarray): Validation data
        y_val (ndarray): Simulated ground truth labels for validation set
    """
    np.random.seed(seed)
    y = np.zeros(X.shape[0])
    anomaly_count = int(anomaly_fraction * X.shape[0])
    y[:anomaly_count] = 1
    np.random.shuffle(y)

    return train_test_split(X, y, test_size=0.3, random_state=seed)

def visualize_detected_image_anomalies(image, anomaly_mask):
    """
    Overlays anomalies on the image using red color.

    Args:
        image (ndarray): Original image (H, W, 3)
        anomaly_mask (ndarray): Boolean array of shape (H*W,) indicating anomalies
    """
    height, width, _ = image.shape
    anomaly_overlay = np.copy(image).reshape(-1, 3)
    anomaly_overlay[anomaly_mask] = [1, 0, 0]  # Red for anomalies
    overlay_img = anomaly_overlay.reshape(height, width, 3)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Anomalies Highlighted")
    plt.imshow(overlay_img)
    plt.axis('off')
    plt.show()