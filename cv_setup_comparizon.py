from data.data import build_models, load_data
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import tensorflow as tf

if __name__ == "__main__":
    x, y = load_data()
    x_train, x_, y_train, y_ = train_test_split(
        x, y, test_size=0.40, random_state=1)
    x_cv, x_test, y_cv, y_test = train_test_split(
        x_, y_, test_size=0.50, random_state=1)

    del x_, y_

    degree = 1
    poly = PolynomialFeatures(degree, include_bias=False)
    X_train_mapped = poly.fit_transform(x_train)
    X_cv_mapped = poly.transform(x_cv)
    X_test_mapped = poly.transform(x_test)

    scaler = StandardScaler()
    X_train_mapped_scaled = scaler.fit_transform(X_train_mapped)
    X_cv_mapped_scaled = scaler.transform(X_cv_mapped)
    X_test_mapped_scaled = scaler.transform(X_test_mapped)

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
            X_train_mapped_scaled, y_train,
            epochs=300,
            verbose=0
        )

        yhat = model.predict(X_train_mapped_scaled)
        train_mse = mean_squared_error(y_train, yhat) / 2
        nn_train_mses.append(train_mse)

        yhat = model.predict(X_cv_mapped_scaled)
        cv_mse = mean_squared_error(y_cv, yhat) / 2
        nn_cv_mses.append(cv_mse)

    print("RESULTS:")
    for model_num in range(len(nn_train_mses)):
        print(
            f"Model {model_num+1}: Training MSE: {nn_train_mses[model_num]:.2f}, " +
            f"CV MSE: {nn_cv_mses[model_num]:.2f}"
        )
