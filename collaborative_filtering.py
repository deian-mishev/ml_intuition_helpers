import numpy as np
import tensorflow as tf
from tensorflow import keras

from data.data import load_movies, load_params, load_ratings

def normalize_ratings(ratings, rated):
    """
    Normalize ratings by subtracting the mean rating for each movie.
    Args:
      ratings (ndarray): Original rating matrix
      rated   (ndarray): Binary mask (1 if rated)
    Returns:
      ratings_norm (ndarray): Normalized ratings
      ratings_mean (ndarray): Mean rating per movie
    """
    ratings_mean = (np.sum(ratings * rated, axis=1) / (np.sum(rated, axis=1) + 1e-12)).reshape(-1, 1)
    ratings_norm = ratings - ratings_mean * rated
    return ratings_norm, ratings_mean

def collaborative_cost_fn(movie_features, user_prefs, user_bias, ratings, rated, lambda_):
    """
    Vectorized cost function for collaborative filtering with TensorFlow compatibility.
    Args:
      movie_features (tf.Variable): Movie feature matrix (num_movies, num_features)
      user_prefs     (tf.Variable): User preference matrix (num_users, num_features)
      user_bias      (tf.Variable): User bias vector (1, num_users)
      ratings        (ndarray): Rating matrix (num_movies, num_users)
      rated          (ndarray): Binary indicator for ratings
      lambda_        (float): Regularization parameter
    Returns:
      cost (tf.Tensor): Scalar cost
    """
    pred = tf.linalg.matmul(movie_features, tf.transpose(user_prefs)) + user_bias
    error = (pred - ratings) * rated
    cost = 0.5 * tf.reduce_sum(tf.square(error)) + 0.5 * lambda_ * (tf.reduce_sum(tf.square(movie_features)) + tf.reduce_sum(tf.square(user_prefs)))
    return cost

if __name__ == "__main__":
    # Load parames
    movie_features_init, user_prefs_init, user_bias_init, num_movies, num_features, num_users = load_params()
    ratings, rated = load_ratings()
    movie_list, movie_df = load_movies()

    # Initialize custom user rating
    my_ratings = np.zeros(num_movies)
    my_ratings[2700] = 5
    my_ratings[2609] = 2
    my_ratings[929]  = 5
    my_ratings[246]  = 5
    my_ratings[2716] = 3
    my_ratings[1150] = 5
    my_ratings[382]  = 2
    my_ratings[366]  = 5
    my_ratings[622]  = 5
    my_ratings[988]  = 3
    my_ratings[2925] = 1
    my_ratings[2937] = 1
    my_ratings[793]  = 5
    my_rated = [i for i in range(len(my_ratings)) if my_ratings[i] > 0]

    # Load ratings and insert user's data
    ratings = np.c_[my_ratings, ratings]
    rated   = np.c_[(my_ratings != 0).astype(int), rated]

    # Normalize the dataset
    ratings_norm, ratings_mean = normalize_ratings(ratings, rated)

    # Set up model variables
    num_movies, num_users = ratings.shape
    num_features = 100
    tf.random.set_seed(1234)

    # Tell TF we want to update this W's
    user_prefs = tf.Variable(tf.random.normal((num_users, num_features), dtype=tf.float64), name='user_prefs')
    movie_features = tf.Variable(tf.random.normal((num_movies, num_features), dtype=tf.float64), name='movie_features')
    user_bias = tf.Variable(tf.random.normal((1, num_users), dtype=tf.float64), name='user_bias')

    optimizer = keras.optimizers.Adam(learning_rate=0.1)
    lambda_ = 1.0
    iterations = 400

    for step in range(iterations):
        with tf.GradientTape() as tape:
            cost = collaborative_cost_fn(movie_features, user_prefs, user_bias, ratings_norm, rated, lambda_)
        grads = tape.gradient(cost, [movie_features, user_prefs, user_bias])
        optimizer.apply_gradients(zip(grads, [movie_features, user_prefs, user_bias]))

        if step % 20 == 0:
            print(f"Training loss at iteration {step}: {cost:.2f}")

    # Predict and de-normalize
    predictions = np.matmul(movie_features.numpy(), user_prefs.numpy().T) + user_bias.numpy()
    final_predictions = predictions + ratings_mean
    my_predictions = final_predictions[:, 0]

    # Display top predictions for unrated movies
    top_indices = tf.argsort(my_predictions, direction='DESCENDING').numpy()

    print("\nTop movie recommendations:\n")
    rec_count = 0
    for i in top_indices:
        if i not in my_rated:
            print(f"Predicting rating {my_predictions[i]:.2f} for movie {movie_list[i]}")
            rec_count += 1
        if rec_count >= 17:
            break

    print('\nOriginal vs Predicted ratings:\n')
    for i in my_rated:
        print(f'Original {my_ratings[i]}, Predicted {my_predictions[i]:.2f} for {movie_list[i]}')