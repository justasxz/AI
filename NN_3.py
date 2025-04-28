import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, Input
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import matplotlib.pyplot as plt

# =============================================================================
# 1) CONFIGURATION & REPRODUCIBILITY
# =============================================================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
# TF2 runs in eager mode by default; the placeholder warning you saw is internal
# and can be ignored unless you’re mixing in TF1 graph code.

# =============================================================================
# 2) LOAD & PREPROCESS MNIST
# =============================================================================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# Scale to [0,1] for stable gradients
x_train = (x_train.astype('float32') / 255.0)[..., np.newaxis]  # → (N,28,28,1)
x_test  = (x_test.astype('float32')  / 255.0)[..., np.newaxis]

# =============================================================================
# 3) BUILD A RESNET-BASED MODEL FOR MNIST
# =============================================================================

# 3.1 Input: grayscale 28×28 images
inputs = Input(shape=(28, 28, 1), name='input_gray')

# 3.2 Convert to RGB by channel–stacking so ResNet can consume 3 channels
#    (ResNet was architected for color images.)
x = layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x),
                  name='to_rgb')(inputs)

# 3.3 Resize to at least 32×32 so the downsampling blocks in ResNet50V2 still work
#     Note the fix: use layers.Resizing, not layers.experimental.preprocessing.Resizing
x = layers.Resizing(32, 32, name='resize')(x)

# 3.4 Apply ResNet’s standard preprocessing (zero-centering + scaling)
x = layers.Lambda(preprocess_input, name='resnet_preproc')(x)

# 3.5 Load ResNet50V2 without its top classification head
#     weights=None trains from scratch; use 'imagenet' to fine-tune pre-trained weights
base = ResNet50V2(include_top=False,
                  weights=None,
                  input_tensor=x,
                  name='ResNet50V2_backbone')

# 3.6 GlobalAveragePooling collapses H×W spatial dims → a feature vector per sample
x = layers.GlobalAveragePooling2D(name='gap')(base.output)

# 3.7 A small dense “head” for our 10-digit classification
x = layers.Dense(64, activation='relu', name='head_dense')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)

# Assemble the model
resnet_model = models.Model(inputs=inputs, outputs=outputs, name='ResNet_MNIST')

# =============================================================================
# 4) COMPILE
# =============================================================================
resnet_model.compile(
    optimizer=tf.keras.optimizers.Adam(),                # adaptive LR
    loss='sparse_categorical_crossentropy',              # integer labels
    metrics=['accuracy']
)

print(resnet_model.summary())

# =============================================================================
# 5) TRAIN
# =============================================================================
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

history = resnet_model.fit(
    x_train, y_train,
    validation_split=0.1,   # hold out 10% for validation
    batch_size=128,
    epochs=10,
    callbacks=[early_stop],
    verbose=2
)

# =============================================================================
# 6) EVALUATE
# =============================================================================
test_loss, test_acc = resnet_model.evaluate(x_test, y_test, verbose=0)
print(f"ResNet50V2 on MNIST — Test Accuracy: {test_acc:.4f}")

# =============================================================================
# 7) SHOW A SAMPLE
# =============================================================================
sample = x_test[0]  # one 28×28×1 image
plt.imshow(sample.squeeze(), cmap='gray')
plt.title(f"True label: {y_test[0]}")
plt.axis('off')
plt.show()
