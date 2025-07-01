from supervised_learning.linear_regression import gradient_descent, compute_cost, compute_gradient, compute_reg_gradient
import numpy as np
import matplotlib.pyplot as plt

def generate_polynomial_data(m, degree, noise_std=15.0, seed=0):
    """
    Generates synthetic data with polynomial relationship y = a0 + a1*x + a2*x^2 + ... + an*x^n + noise

    Args:
        m (int): Number of data points
        degree (int): Highest degree of polynomial
        noise_std (float): Standard deviation of Gaussian noise
        seed (int): Random seed for reproducibility

    Returns:
        X (ndarray (m,1)): Input feature values
        y (ndarray (m,)): Target values
        true_coefs (ndarray (degree+1,)): Coefficients used to generate y
    """
    np.random.seed(seed)
    X_orig = 2 * np.random.rand(m)
    
    # Force high-degree coefficients to be large
    true_coefs = np.array([np.random.randn() * (1 if i < 2 else 10) for i in range(degree + 1)])

    y = np.zeros(m)
    for i, coef in enumerate(true_coefs):
        y += coef * (X_orig ** i)
    # y = np.clip(y, -1e4, 1e4)
    y += np.random.randn(m) * noise_std
    
    return X_orig, y, true_coefs

def prepare_poly_features(x, degree):
    """
    Generates polynomial features [x, x^2, ..., x^degree]
    Args:
        x (ndarray): shape (m,) or (m,1)
        degree (int): Maximum degree of polynomial
    Returns:
        X_poly (ndarray): shape (m, degree) with polynomial features
    """
    x = x.reshape(-1, 1)
    return np.hstack([x ** d for d in range(1, degree + 1)])

def zscore_normalize_features(X):
    """
    Standardizes features by removing the mean and scaling to unit variance
    Args:
        X (ndarray): shape (m, n) - Input features
    Returns:
        X_norm (ndarray): Normalized features
        mu (ndarray): Mean of each feature
        sigma (ndarray): Std of each feature
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

if __name__ == "__main__":
    # Parameters
    m = 5
    degree = 14
    noise_std = 1000

    # Generate data and coefficients
    X_orig, y, true_coefs = generate_polynomial_data(m, degree, noise_std)

    # Prepare polynomial features (up to degree)
    X_poly = prepare_poly_features(X_orig, degree)

    # Normalize polynomial features
    # X_poly_norm, mu, sigma = zscore_normalize_features(X_poly)
    X_poly_norm = X_poly

    # Initialize weights and bias
    w_init = np.zeros(X_poly_norm.shape[1])
    b_init = 0.0
    alpha = 1e-5
    num_iters = 100000

    # Run gradient descent
    w, b, J_hist, p_hist = gradient_descent(
        X_poly_norm, y, w_init, b_init, alpha, num_iters, compute_cost, compute_gradient,
        (lambda w, X: compute_reg_gradient(w, X, 10))
    )
    print(f"Gradient descent result:\n w = {w}\n b = {b:.3f}")

    # Plot predictions
    X_plot = np.linspace(X_orig.min(), X_orig.max(), 300)
    X_plot_poly = prepare_poly_features(X_plot, degree)
    y_pred = X_plot_poly @ w + b

    plt.figure(figsize=(8, 6))
    plt.scatter(X_orig, y, label="Training data", color='blue')
    plt.plot(X_plot, y_pred, label="Overfit model", color='red')
    plt.legend()
    plt.title("Intentional Overfitting (No Normalization)")
    plt.grid(True)
    plt.show()