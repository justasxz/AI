# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import Dense
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# # from sklearn.datasets import load_iris


# # iris = load_iris()
# # X = iris.data
# # y = iris.target


# # model = Sequential() # neuronai eina vienos po kito, todel mes naudojam sequential
# # model.add(Dense(8, activation='relu', input_shape=(X.shape[1],)))
# # model.add(Dense(8, activation='relu'))
# # model.add(Dense(3, activation='softmax'))
# # # if we want to set learning rate, we can do it like this:
# # from tensorflow.keras.optimizers import Adam

# # adam = Adam(learning_rate=0.01) # learning rate - kiek mes norim, kad modelis keistusi (kiekviena karta kai jis keiciasi, jis keiciasi tiek, kiek mes jam pasakom)
# # model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy']) # adam yra geriausias optimizeris, ji dazniausiai ir naudojam (veliau prasiplesiu apie kitus, bet jie labiau naudojami nisinese situacijose)
# # # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # adam yra geriausias optimizeris, ji dazniausiai ir naudojam (veliau prasiplesiu apie kitus, bet jie labiau naudojami nisinese situacijose)
# # # validation

# # model.fit(X, y, epochs=10, batch_size=10, verbose=2, validation_split=0.1) # epochs - kiek kartu perziurim visus duomenis, batch_size - kiek duomenu vienu metu perziurim (kiekvienam epochui)

# import optuna
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.callbacks import TensorBoard
# from tensorflow.keras import layers
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_iris
# from optuna.integration import TFKerasPruningCallback

# # --- data prep (example) ---
# iris = load_iris()
# X, y = iris.data, iris.target

# X_train, X_val, y_train, y_val = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# input_dim   = X_train.shape[1]
# num_classes = len(np.unique(y_train))

# # --- simplified Optuna objective ---
# def objective(trial):
#     # 1) hyperparameters to tune
#     units     = trial.suggest_int("units", 32, 256, step=32) # 32 ,64, 96, 128, 160, 192, 224, 256
#     # activation = trial.suggest_categorical("activation", ["relu", "tanh"])
#     # learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
#     # optimizer = trial.suggest_categorical("optimizer", ["adam", "sgd"])
#     # dropout   = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
#     batch_sz  = trial.suggest_categorical("batch_size", [16, 32, 64])

#     # 2) build model
#     model = keras.Sequential([
#         layers.Input(shape=(input_dim,)),
#         layers.Dense(units, activation="relu"),
#         layers.Dropout(0.2),
#         layers.Dense(num_classes, activation="softmax"),
#     ])

#     # 3) compile with a fixed optimizer
#     model.compile(
#         optimizer="adam",
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"]
#     )

#     # 4) fit with pruning callback
#     callbacks = [
#         tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
#         TFKerasPruningCallback(trial, "val_loss"),
#         TensorBoard(log_dir="logs/fit/" + str(trial.number), histogram_freq=1, write_graph=True, write_images=True)
#     ]
    
#     model.fit(
#         X_train, y_train,
#         validation_data=(X_val, y_val),
#         epochs=30,
#         batch_size=batch_sz,
#         callbacks=callbacks,
#         verbose=0
#     )

#     # 5) evaluate
#     _, val_acc = model.evaluate(X_val, y_val, verbose=0)
#     return val_acc

# # --- run the study ---
# study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
# study.optimize(objective, n_trials=25)

# print("Best params:", study.best_trial.params)
# print("Best val accuracy: {:.4f}".format(study.best_value))


# def keiciam_sarasa(s):
#     s = s.append(4) 
#     # s = s + [4]
#     return s

# mano = [1, 2, 3]
# keiciam_sarasa(mano)
# print(mano)





import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, Input
import matplotlib.pyplot as plt

# =============================================================================
# 1. CONFIGURATION & REPRODUCIBILITY
# =============================================================================

# 1.1 Set a global seed for numpy, Python, and TensorFlow to ensure reproducible results
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Eager execution is enabled by default in TF2, making debugging and intuitive model building easier.

# =============================================================================
# 2. LOAD & PREPROCESS DATA
# =============================================================================

def load_and_preprocess_mnist():
    """
    Loads MNIST, scales pixel values to [0,1], and returns:
    - flat images for FFNN: shape (batch, 28, 28)
    - channel-extended images for CNN: shape (batch, 28, 28, 1)
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Convert to float32 then normalize to [0,1] for numerical stability during training
    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32') / 255.0

    # Add a channels dimension (required for Conv2D): (28, 28) → (28, 28, 1)
    x_train_cnn = np.expand_dims(x_train, -1)
    x_test_cnn  = np.expand_dims(x_test, -1)

    return (x_train, y_train), (x_test, y_test), (x_train_cnn, x_test_cnn)

# Load and split the data
(x_train, y_train), (x_test, y_test), (x_train_cnn, x_test_cnn) = load_and_preprocess_mnist()

# =============================================================================
# 3. BUILD MODEL FACTORIES
# =============================================================================
def build_ffnn(input_shape=(28,28), n_classes=10):
    """
    Constructs a simple FFNN:
    - Flatten: to convert 2D images into 1D vectors
    - Dense layers: learn non-linear combinations of pixels
    - softmax output: for 10-class classification
    """
    model = models.Sequential(name='FFNN')
    model.add(layers.Flatten(input_shape=input_shape, name='flatten'))
    model.add(layers.Dense(128, activation='relu', name='dense_128'))
    model.add(layers.Dense(64, activation='relu', name='dense_64'))
    model.add(layers.Dense(n_classes, activation='softmax', name='softmax_output'))
    return model

def build_cnn(input_shape=(28,28,1), n_classes=10):
    """
    Constructs a simple CNN:
    - Conv2D → ReLU: learns local edge detectors
    - MaxPooling: reduces spatial dims, adds translation invariance
    - Additional Conv2D layers: learn higher-level motifs
    - Flatten + Dense: combine extracted features for classification
    """
    model = models.Sequential(name='CNN')
    model.add(layers.Conv2D(32, (2,2), activation='relu', name='conv1',
                            input_shape=input_shape)) # 32 filters, 3x3 the size of each filter
    model.add(layers.MaxPooling2D((2,2), name='pool1')) # 2x2 pooling reduces spatial dims, maxpooling, takes the max value in each 2x2 block, 
    #minpooling would take the min value, averagepooling would take the average value
    model.add(layers.Conv2D(64, (2,2), activation='relu', name='conv2'))
    model.add(layers.MaxPooling2D((2,2), name='pool2')) # convolutional and pooling layers where used twice to learn more complex features
    model.add(layers.Flatten(name='flatten'))
    model.add(layers.Dense(64, activation='relu', name='dense_64'))
    model.add(layers.Dense(64, activation='relu', name='dense_64_2'))
    model.add(layers.Dense(n_classes, activation='softmax', name='softmax_output'))
    return model

# Create instances of both models
ffnn = build_ffnn()
cnn  = build_cnn()

# Print summary to inspect layer shapes and parameter counts
print(ffnn.summary())
print(cnn.summary())

# =============================================================================
# 4. COMPILE MODELS
# =============================================================================

# Use Adam optimizer (adaptive LR), sparse_categorical_crossentropy for integer labels,
# and track accuracy
for model in (ffnn, cnn):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# =============================================================================
# 5. CREATE CALLBACKS
# =============================================================================

# EarlyStopping: halt training when validation loss plateaus to prevent overfitting
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,              # wait this many epochs before stopping
    restore_best_weights=True,
    verbose=1,
)

# =============================================================================
# 6. TRAINING
# =============================================================================

BATCH_SIZE = 128
EPOCHS     = 10

print("\n>>> Training FFNN")
history_ffnn = ffnn.fit(
    x_train, y_train,
    validation_split=0.1,     # hold out 10% of training for validation
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stop],
    verbose=2,
)

print("\n>>> Training CNN")
history_cnn = cnn.fit(
    x_train_cnn, y_train,
    validation_split=0.1,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stop],
    verbose=2,
)
# # save to file
# ffnn.save('ffnn_model.h5')
# # load from file
# ffnn = tf.keras.models.load_model('ffnn_model.h5')
# =============================================================================
# 7. EVALUATION
# =============================================================================

# Evaluate final performance on the unseen test set
test_loss_ffnn, test_acc_ffnn = ffnn.evaluate(x_test, y_test, verbose=0)
test_loss_cnn,  test_acc_cnn  = cnn.evaluate(x_test_cnn, y_test, verbose=0)
print(f"\nFFNN Test Accuracy: {test_acc_ffnn:.4f}")
print(f"CNN  Test Accuracy: {test_acc_cnn:.4f}")

# =============================================================================
# 8. PLOT RESULTS
# =============================================================================

def plot_validation_accuracy(h1, h2):
    """Overlay validation accuracy curves for FFNN vs CNN."""
    plt.figure(figsize=(8,5))
    plt.plot(h1.history['val_accuracy'], label='FFNN Val Acc')
    plt.plot(h2.history['val_accuracy'], label='CNN Val Acc')
    plt.title('Validation Accuracy: FFNN vs CNN')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_validation_accuracy(history_ffnn, history_cnn)

# =============================================================================
# 9. VISUALIZE FEATURE MAPS (WITH FIX)
# =============================================================================

# Choose a sample test image (28x28x1)
sample_img = x_test_cnn[0]

# Show the original digit
plt.figure()
plt.imshow(sample_img.squeeze(), cmap='gray')
plt.title('Sample Input')
plt.axis('off')
plt.show()

# --- FIX: Instead of relying on cnn.input (which may not be built),
#     we use the functional API to extract feature maps:
# 1. Grab the trained 'conv1' layer from the CNN
conv1_layer = cnn.get_layer('conv1')

# 2. Create a new Input tensor matching the conv1 input shape
inp = Input(shape=(28, 28, 1), name='feature_input')

# 3. Apply the conv1 layer to this Input
feature_maps_tensor = conv1_layer(inp)

# 4. Build a lightweight Model from inp → conv1 activations
feature_extractor = models.Model(inputs=inp,
                                 outputs=feature_maps_tensor,
                                 name='feature_extractor')

# 5. Compute the activations for our sample image
feature_maps = feature_extractor.predict(sample_img[np.newaxis, ...])

# 6. Plot first 9 feature maps to illustrate learned filters
plt.figure(figsize=(6,6))
for i in range(9):
    ax = plt.subplot(3, 3, i+1)
    plt.imshow(feature_maps[0, :, :, i], cmap='gray')
    plt.title(f'Filter {i+1}')
    plt.axis('off')
plt.suptitle('First 9 Feature Maps from conv1')
plt.tight_layout()
plt.show()
