import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import data  # Assuming the data module is available

# Load the dataset
train_data, train_labels, test_data, test_labels = data.load_all_data('data')

# Function to apply diffusion as a Markov chain
def apply_markov_noise(data, step_size=0.1, steps=10):
    noisy_data = data.copy()
    for _ in range(steps):
        noisy_data += np.random.normal(scale=step_size, size=noisy_data.shape)
    return noisy_data

# Apply Markov chain noise to the data
noisy_train_data = apply_markov_noise(train_data, step_size=0.1, steps=10)
noisy_test_data = apply_markov_noise(test_data, step_size=0.1, steps=10)

# Create a simple neural network for denoising
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(64,)),
    layers.Dense(64),
    layers.Dense(64)  # Output layer with the same size as the input
])

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(noisy_train_data, train_data, epochs=100, batch_size=32)

# Denoise the test data
denoised_test_data = model.predict(noisy_test_data)

# Function to plot images
def plot_images(images, labels, num_images=5, title=""):
    fig, axes = plt.subplots(1, num_images, figsize=(10, 2))
    for i, ax in enumerate(axes):
        ax.imshow(images[i].reshape(8, 8), cmap='gray')
        ax.set_title(f'Label: {int(labels[i])}')
        ax.axis('off')
    plt.suptitle(title)
    plt.show()

# Visualize original, noisy, and denoised images
plot_images(test_data, test_labels, num_images=5, title="Original Images")
plot_images(noisy_test_data, test_labels, num_images=5, title="Noisy Images")
plot_images(denoised_test_data, test_labels, num_images=5, title="Denoised Images")
