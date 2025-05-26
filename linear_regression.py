import math
import numpy as np
import matplotlib.pyplot as plt

def compute_reg_gradient(w, X, lambda_=1):
    """
    Computes the L2 regularization cost over all weights

    Args:
      w (ndarray (n,)): model parameters  
      X (ndarray (m,n)): Data matrix with m examples and n features
      lambda_ (float) : regularization strength

    Returns:
      reg_gradient (ndarray (n,)): gradient of the regularization term with respect to w
    """
    m = X.shape[0]
    reg_gradient = (lambda_ / m) * w
    return reg_gradient

def compute_cost(X, y, w, b):
    """
    Computes the cost for linear regression 
    Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,)): target values
        w,b (scalar)    : model parameters  
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
        to fit the data points in x and y
    """
    m = X.shape[0]
    f_wb = X @ w + b
    cost = (1 / (2 * m)) * np.sum((f_wb - y) ** 2)
    return cost


def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression 
    Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,)): target values
        w,b (scalar)    : model parameters  
    Returns
        dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
        dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
    """
    m = X.shape[0]
    f_wb = X @ w + b  # shape (m,)
    error = f_wb - y  # shape (m,)
    dj_dw = (X.T @ error) / m  # shape (n,)
    dj_db = np.sum(error) / m  # scalar
    return dj_dw, dj_db


def gradient_descent(X, y, w, b, alpha, num_iters,
                     cost_function,
                     gradient_function, regularization=None):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha

    Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,))  : target values
        w,b (scalar): initial values of model parameters  
        alpha (float):     Learning rate
        num_iters (int):   number of iterations to run gradient descent
        cost_function:     function to call to produce cost
        gradient_function: function to call to produce gradient

    Returns:
        w (scalar): Updated value of parameter after running gradient descent
        b (scalar): Updated value of parameter after running gradient descent
        J_history (List): History of cost values
        p_history (list): History of parameters [w,b] 
    """
    reg_value = regularization if regularization is not None else (lambda w, X: 0)

    J_history = []
    p_history = []

    for i in range(num_iters):
        dj_dw_loss, dj_db = gradient_function(X, y, w, b)
        dj_dw = dj_dw_loss + reg_value(w, X)
        # Update parameters
        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i < 100000:
            J_history.append(cost_function(X, y, w, b))
            p_history.append([w, b])
        if i % max(1, math.ceil(num_iters / 10)) == 0:
            weights_str = ", ".join(
                [f"w[{j}] = {wi:.3f}" for j, wi in enumerate(w)])
            print(f"Weights: {weights_str} | Bias: b = {b:.3f}")

    return w, b, J_history, p_history


if __name__ == "__main__":
    np.random.seed(0)
    m = 100
    X = 2 * np.random.rand(m, 1)
    y = 4 + 3 * X[:, 0] + np.random.randn(m)

    b_init = 0.0
    w_init = np.zeros(X.shape[1])
    alpha = 1.0e-2
    num_iters = 20000

    w, b, J_hist, p_hist = gradient_descent(
        X, y, w_init, b_init, alpha, num_iters, compute_cost, compute_gradient
    )
    fig, (ax1, ax2) = plt.subplots(
        1, 2, constrained_layout=True, figsize=(12, 4))
    ax1.plot(J_hist[:100])
    ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
    ax1.set_title("Cost vs. iteration(start)")
    ax2.set_title("Cost vs. iteration (end)")
    ax1.set_ylabel('Cost')
    ax2.set_ylabel('Cost')
    ax1.set_xlabel('iteration step')
    ax2.set_xlabel('iteration step')
    plt.show()

    # Make predictions on new data
    x_test = np.array([0, 1, 2])
    y_pred = w * x_test + b
    for x_val, y_val in zip(x_test, y_pred):
        print(f"Prediction: for x = {x_val}, predicted y = {y_val:.2f}")

    # Plot the training data and the linear regression line
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], y, color='blue', label='Training data')

    # Predictions for line (sort X for smooth line)
    X_plot = np.linspace(X.min(), X.max(), 100)
    y_plot = w[0] * X_plot + b

    plt.plot(X_plot, y_plot, color='red',
             label=f'Linear regression line: y = {w[0]:.2f}x + {b:.2f}')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.grid(True)
    plt.show()
