from data.data import build_models, load_data
from sklearn.metrics import mean_squared_error

import tensorflow as tf

if __name__ == "__main__":
    x_train, x_cv, x_test, y_train, y_cv, y_test = load_data()

    nn_train_mses = []
    nn_cv_mses = []
    nn_models = build_models()

    for model in nn_models:

        model.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        )

        print(f"Training {model.name}...")

        model.fit(
            x_train, y_train,
            epochs=300,
            verbose=0
        )

        yhat = model.predict(x_train)
        train_mse = mean_squared_error(y_train, yhat) / 2
        nn_train_mses.append(train_mse)

        yhat = model.predict(x_cv)
        cv_mse = mean_squared_error(y_cv, yhat) / 2
        nn_cv_mses.append(cv_mse)

    print("RESULTS:")
    for model_num in range(len(nn_train_mses)):
        print(
            f"Model {model_num+1}: Training MSE: {nn_train_mses[model_num]:.2f}, " +
            f"CV MSE: {nn_cv_mses[model_num]:.2f}"
        )
