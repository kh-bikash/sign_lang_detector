
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

print("TensorFlow Version:", tf.__version__)

# --- Load Processed Data ---
print("Loading preprocessed data...")
try:
    X_train = np.load('processed_data/X_train.npy')
    y_train = np.load('processed_data/y_train.npy')
    X_test = np.load('processed_data/X_test.npy')
    y_test = np.load('processed_data/y_test.npy')
except FileNotFoundError:
    print("\nError: Processed data not found!")
    print("Please run 'preprocess_kaggle_data.py' first to generate the data.")
    exit()

print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")


# --- Build the CNN Model ---
print("Building the Convolutional Neural Network (CNN) model...")
model = Sequential([
    # First convolution layer
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Second convolution layer
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Flatten the results to feed into a dense layer
    Flatten(),

    # Dense layer
    Dense(128, activation='relu'),
    Dropout(0.5),

    # Output layer
    # 25 classes, so we use 'softmax' for multi-class classification
    Dense(25, activation='softmax')
])

# --- Compile the Model ---
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# --- Train the Model ---
print("\nStarting model training...")
# An epoch is one complete pass through the entire training dataset.
# We'll do 10 passes to let the model learn effectively.
history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))

# --- Evaluate the Model ---
print("\nEvaluating model performance...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc:.4f}')


# --- Save the Trained Model ---
print("Saving the trained model to 'asl_model.h5'...")
model.save('asl_model.h5')

print("\nModel training complete and model saved successfully!")
