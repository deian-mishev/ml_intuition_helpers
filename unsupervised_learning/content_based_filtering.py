import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from data.data import load_data

def scale_data(user_data, item_data, y_data):
    """
    Scales user, item, and target data using standard and min-max scaling.
    Args:
      user_data (ndarray): Raw user feature data.
      item_data (ndarray): Raw item feature data.
      y_data (ndarray): Target ratings.
    Returns:
      user_scaled (ndarray): Scaled user data.
      item_scaled (ndarray): Scaled item data.
      y_scaled (ndarray): Scaled target data in range [-1, 1].
      user_scaler (StandardScaler): Fitted user scaler.
      item_scaler (StandardScaler): Fitted item scaler.
      target_scaler (MinMaxScaler): Fitted target scaler.
    """
    user_scaler = StandardScaler()
    item_scaler = StandardScaler()
    target_scaler = MinMaxScaler(feature_range=(-1, 1))

    user_scaled = user_scaler.fit_transform(user_data)
    item_scaled = item_scaler.fit_transform(item_data)
    y_scaled = target_scaler.fit_transform(y_data.reshape(-1, 1))

    return user_scaled, item_scaled, y_scaled, user_scaler, item_scaler, target_scaler


def build_model(num_user_features, num_item_features, num_outputs=32):
    """
    Builds a neural collaborative filtering model with dot-product output.
    Args:
      num_user_features (int): Number of features for user input.
      num_item_features (int): Number of features for item input.
      num_outputs (int): Size of the embedding output vector.
    Returns:
      model (tf.keras.Model): Compiled collaborative filtering model.
    """
    user_NN = keras.Sequential([
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_outputs)
    ])

    item_NN = keras.Sequential([
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_outputs)
    ])

    input_user = keras.layers.Input(shape=(num_user_features,))
    input_item = keras.layers.Input(shape=(num_item_features,))

    vu = user_NN(input_user)
    vm = item_NN(input_item)

    vu = keras.layers.Lambda(lambda x: tf.linalg.l2_normalize(x, axis=1))(vu)
    vm =keras.layers. Lambda(lambda x: tf.linalg.l2_normalize(x, axis=1))(vm)

    output = keras.layers.Dot(axes=1)([vu, vm])
    model = keras.Model([input_user, input_item], output)
    return model


def get_user_vecs(user_id, user_train, item_vecs, user_to_genre):
    """
    Constructs a matrix of user vectors and corresponding ratings for prediction.
    Args:
      user_id (int): The user ID to generate vectors for.
      user_train (ndarray): Scaled user training data.
      item_vecs (ndarray): Raw item matrix with movie IDs.
      user_to_genre (dict): Mapping of user IDs to movie ratings.
    Returns:
      user_vecs (ndarray): User matrix with rows repeated to match item_vecs.
      y (ndarray): Ratings for rated movies, 0 otherwise.
    """
    if user_id not in user_to_genre:
        print("Error: unknown user id")
        return None

    user_vec_found = False
    for i in range(len(user_train)):
        if user_train[i, 0] == user_id:
            user_vec = user_train[i]
            user_vec_found = True
            break

    if not user_vec_found:
        print("Error: did not find user_id in user_train")
        return None

    num_items = len(item_vecs)
    user_vecs = np.tile(user_vec, (num_items, 1))

    y = np.zeros(num_items)
    for i in range(num_items):
        movie_id = item_vecs[i, 0]
        rating = user_to_genre[user_id]['movies'].get(movie_id, 0)
        y[i] = rating

    return user_vecs, y


if __name__ == "__main__":
    item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre = load_data()

    u_s = 3  # Start index for user features (excluding ID, etc.)
    i_s = 1  # Start index for item features (excluding ID)

    num_user_features = user_train.shape[1] - u_s
    num_item_features = item_train.shape[1] - i_s

    print(f"Number of training vectors: {len(item_train)}")

    user_train, item_train, y_train, scalerUser, scalerItem, scalerTarget = scale_data(user_train, item_train, y_train)

    # Train/test split
    item_train, item_test = train_test_split(item_train, train_size=0.80, shuffle=True, random_state=1)
    user_train, user_test = train_test_split(user_train, train_size=0.80, shuffle=True, random_state=1)
    y_train, y_test       = train_test_split(y_train,    train_size=0.80, shuffle=True, random_state=1)

    print(f"movie/item training data shape: {item_train.shape}")
    print(f"movie/item test data shape: {item_test.shape}")

    # Build and train model
    tf.random.set_seed(1)
    model = build_model(num_user_features, num_item_features)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss=keras.losses.MeanSquaredError()
    )

    model.fit([user_train[:, u_s:], item_train[:, i_s:]], y_train, epochs=30)
    model.evaluate([user_test[:, u_s:], item_test[:, i_s:]], y_test)


