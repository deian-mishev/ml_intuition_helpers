from linear_regression import gradient_descent, compute_cost, compute_gradient
import numpy as np
import matplotlib.pyplot as plt

def generate_polynomial_data(m, degree, noise_std=1.0, seed=0):
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
    # Generate input X uniformly between 0 and 2
    X_orig = 2 * np.random.rand(m)
    
    # Random coefficients for polynomial of degree 'degree' (including intercept)
    true_coefs = np.random.randn(degree + 1) * 2
    
    # Compute y = a0 + a1*x + a2*x^2 + ... + noise
    y = np.zeros(m)
    for i, coef in enumerate(true_coefs):
        y += coef * (X_orig ** i)
    y += np.random.randn(m) * noise_std  # add noise
    
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
    m = 100
    degree = 10
    noise_std = 0.8

    # Generate data and coefficients
    X_orig, y, true_coefs = generate_polynomial_data(m, degree, noise_std)

    # Prepare polynomial features (up to degree)
    X_poly = prepare_poly_features(X_orig, degree)

    # Normalize polynomial features
    X_poly_norm, mu, sigma = zscore_normalize_features(X_poly)

    # Initialize weights and bias
    w_init = np.zeros(X_poly.shape[1]) 
    b_init = 0.0
    alpha = 1.0e-2
    num_iters = 20000

    # Run gradient descent
    w, b, J_hist, p_hist = gradient_descent(
        X_poly_norm, y, w_init, b_init, alpha, num_iters, compute_cost, compute_gradient
    )
    print(f"Gradient descent result:\n w = {w}\n b = {b:.3f}")

    # Plot training loss
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
    ax1.plot(J_hist[:100])
    ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
    ax1.set_title("Cost vs. iteration (start)"); ax2.set_title("Cost vs. iteration (end)")
    ax1.set_ylabel('Cost'); ax2.set_ylabel('Cost')
    ax1.set_xlabel('Iteration'); ax2.set_xlabel('Iteration')
    plt.show()

    # Plot the fitted curve
    X_plot = np.linspace(0, 2, 100)
    X_plot_poly = prepare_poly_features(X_plot, degree)
    X_plot_poly_norm = (X_plot_poly - mu) / sigma
    y_pred = X_plot_poly_norm @ w + b

    plt.figure(figsize=(8, 6))
    plt.scatter(X_orig, y, label="Training data", alpha=0.7)
    plt.plot(X_plot, y_pred, color="red", label="Polynomial fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Polynomial Regression Fit (degree={degree})")
    plt.legend()
    plt.grid(True)
    plt.show()