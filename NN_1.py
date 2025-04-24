# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris


# iris = load_iris()
# X = iris.data
# y = iris.target


# model = Sequential() # neuronai eina vienos po kito, todel mes naudojam sequential
# model.add(Dense(8, activation='relu', input_shape=(X.shape[1],)))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(3, activation='softmax'))
# # if we want to set learning rate, we can do it like this:
# from tensorflow.keras.optimizers import Adam

# adam = Adam(learning_rate=0.01) # learning rate - kiek mes norim, kad modelis keistusi (kiekviena karta kai jis keiciasi, jis keiciasi tiek, kiek mes jam pasakom)
# model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy']) # adam yra geriausias optimizeris, ji dazniausiai ir naudojam (veliau prasiplesiu apie kitus, bet jie labiau naudojami nisinese situacijose)
# # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # adam yra geriausias optimizeris, ji dazniausiai ir naudojam (veliau prasiplesiu apie kitus, bet jie labiau naudojami nisinese situacijose)
# # validation

# model.fit(X, y, epochs=10, batch_size=10, verbose=2, validation_split=0.1) # epochs - kiek kartu perziurim visus duomenis, batch_size - kiek duomenu vienu metu perziurim (kiekvienam epochui)

import optuna
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from optuna.integration import TFKerasPruningCallback

# --- data prep (example) ---
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

input_dim   = X_train.shape[1]
num_classes = len(np.unique(y_train))

# --- simplified Optuna objective ---
def objective(trial):
    # 1) hyperparameters to tune
    units     = trial.suggest_int("units", 32, 256, step=32) # 32 ,64, 96, 128, 160, 192, 224, 256
    # activation = trial.suggest_categorical("activation", ["relu", "tanh"])
    # learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    # optimizer = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    # dropout   = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
    batch_sz  = trial.suggest_categorical("batch_size", [16, 32, 64])

    # 2) build model
    model = keras.Sequential([
        layers.Dense(units, activation="relu", input_shape=(input_dim,)),
        layers.Dense(num_classes, activation="softmax"),
    ])

    # 3) compile with a fixed optimizer
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 4) fit with pruning callback
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        TFKerasPruningCallback(trial, "val_loss"),
        TensorBoard(log_dir="logs/fit/" + str(trial.number), histogram_freq=1, write_graph=True, write_images=True)
    ]
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=batch_sz,
        callbacks=callbacks,
        verbose=0
    )

    # 5) evaluate
    _, val_acc = model.evaluate(X_val, y_val, verbose=0)
    return val_acc

# --- run the study ---
study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
study.optimize(objective, n_trials=25)

print("Best params:", study.best_trial.params)
print("Best val accuracy: {:.4f}".format(study.best_value))

