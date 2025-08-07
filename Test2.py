# Setup
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load and Prepare the Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension (28, 28) → (28, 28, 1)
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Define the CNN model WITH dropout
def build_model_with_dropout():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # 🔸 Dropout layer added
        layers.Dense(10, activation='softmax')
    ])
    return model

# Compile and train the model
model_with_dropout = build_model_with_dropout()
model_with_dropout.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

history_with_dropout = model_with_dropout.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    verbose=2
)

# Evaluate on Test Set
loss_with_dropout, acc_with_dropout = model_with_dropout.evaluate(x_test, y_test, verbose=0)
print(f"Accuracy WITH dropout: {acc_with_dropout:.4f}")

# Plot training vs validation curves
def plot_history(h, label="With Dropout"):
    plt.figure(figsize=(14, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(h.history['val_accuracy'], label=f'{label} - val')
    plt.plot(h.history['accuracy'], '--', label=f'{label} - train')
    plt.title("Validation vs Training Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(h.history['val_loss'], label=f'{label} - val')
    plt.plot(h.history['loss'], '--', label=f'{label} - train')
    plt.title("Validation vs Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Plot results
plot_history(history_with_dropout)
