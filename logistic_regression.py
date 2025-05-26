from linear_regression import gradient_descent, compute_cost
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500)  # clip to prevent overflow
    return 1 / (1 + np.exp(-z))

def compute_cost_logistic_sq_err(X, y, w, b):
    """
    compute sq error cost on logicist data (for negative example only, not used in practice)
    Args:
      X (ndarray): Shape (m,n) matrix of examples with multiple features
      w (ndarray): Shape (n)   parameters for prediction
      b (scalar):              parameter  for prediction
    Returns:
      cost (scalar): cost
    """
    m = X.shape[0]
    z = X @ w + b              # shape (m,)
    f_wb = sigmoid(z)          # shape (m,)
    cost = np.mean((f_wb - y) ** 2) / 2
    return np.squeeze(cost)

def compute_gradient_logistic(X, y, w, b): 
    """
    Computes the gradient for logistic regression 
 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
    Returns
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b. 
    """
    m = X.shape[0]
    z = X @ w + b              # shape (m,)
    f_wb = sigmoid(z)          # shape (m,)
    error = f_wb - y           # shape (m,)
    
    dj_dw = (X.T @ error) / m  # shape (n,)
    dj_db = np.sum(error) / m  # scalar

    return dj_dw, dj_db


if __name__ == "__main__":
    # Parameters
    np.random.seed(0)

    m = 100
    X_orig = 2 * np.random.rand(m, 1)  # shape (m, 1)

    # Generate binary labels for logistic regression with some noise
    # Using a logistic function with some coefficients for generating y
    true_w = np.array([3.0])  # coefficient for x
    true_b = -4.0             # intercept
    linear_combination = X_orig @ true_w + true_b
    prob = sigmoid(linear_combination)
    y = (prob >= 0.5).astype(int)  # binary labels 0 or 1

    # Initialize parameters for logistic regression
    n_features = X_orig.shape[1]
    w_init = np.zeros(n_features)
    b_init = 0.0

    alpha = 0.1
    num_iters = 5000

    # Run gradient descent
    w, b, J_hist, p_hist = gradient_descent(
        X_orig, y, w_init, b_init, alpha, num_iters, compute_cost_logistic_sq_err, compute_gradient_logistic
    )
    print(f"Gradient descent result:\n w = {w}\n b = {b:.3f}")

    # Plot training loss
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))

    ax1.plot(J_hist[:100])
    ax1.set_title("Cost vs. iteration (start)")
    ax1.set_ylabel('Cost')
    ax1.set_xlabel('Iteration')

    ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
    ax2.set_title("Cost vs. iteration (end)")
    ax2.set_ylabel('Cost')
    ax2.set_xlabel('Iteration')

    plt.show()

    # Plot the fitted curve
    X_plot = np.linspace(0, 2, 300).reshape(-1, 1)
    y_prob = sigmoid(X_plot @ w + b)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_orig, y, label="Training data", alpha=0.7)
    plt.plot(X_plot, y_prob, color="red", label="Logistic fit")
    plt.xlabel("x")
    plt.ylabel("Probability")
    plt.title("Logistic Regression Fit")
    plt.legend()
    plt.grid(True)
    plt.show()