import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import data
from tqdm import tqdm
import os


# Define a PyTorch Dataset
class NoisyDataset(Dataset):
    def __init__(self, data, labels, noise_fn):
        self.data = data
        self.labels = labels
        self.noise_fn = noise_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        noisy_data = self.noise_fn(self.data[idx])
        return noisy_data, self.data[idx]


# Apply Markov chain noise to the data
def apply_markov_noise(data, step_size=0.1, steps=10, noise_type="normal"):
    noisy_data = data.copy()
    for _ in range(steps):
        if noise_type == "normal":
            noisy_data += np.random.normal(scale=step_size, size=noisy_data.shape)
        elif noise_type == "laplace":
            # loc makes mean of laplace distribution 0
            noisy_data += np.random.laplace(loc=0, scale=step_size, size=noisy_data.shape)
        else:
            raise ValueError("Invalid noise argument; unsupported noise type.")
    return noisy_data


# Define the neural network
class DenoisingNet(nn.Module):
    def __init__(self):
        super(DenoisingNet, self).__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.fc3(x)


# Denoise the test data
def denoise_data(loader):
    model.eval()
    denoised_data = []
    with torch.no_grad():
        # Wrap test_loader with tqdm for a progress bar
        for noisy_data, _ in tqdm(loader, desc="Denoising Test Data"):
            outputs = model(noisy_data.float())
            denoised_data.append(outputs.cpu().numpy())
    return np.concatenate(denoised_data, axis=0)


def plot_images(images, labels, num_images=10, title="", save_directory="results/mini_mnist/"):
    # Create the directory if it does not exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    fig, axes = plt.subplots(1, num_images, figsize=(20, 2))  # Adjusted for 10 images
    for i, ax in enumerate(axes):
        ax.imshow(images[i].reshape(8, 8), cmap='gray')
        ax.set_title(f'Label: {int(labels[i])}')
        ax.axis('off')
    plt.suptitle(title)

    # Save the figure
    plt.savefig(os.path.join(save_directory, title + ".png"), dpi=300, format='png')
    plt.close(fig)  # Close the figure to free memory


def select_one_image_per_label(data, labels):
    unique_images = []
    unique_labels = []
    for label in range(10):  # Loop through labels 0 to 9
        index = np.where(labels == label)[0][0]  # Find the first occurrence of each label
        unique_images.append(data[index])
        unique_labels.append(labels[index])
    return np.array(unique_images), np.array(unique_labels)


if __name__ == '__main__':
    # Select type of noise
    noise_type = "laplace"

    # Load the dataset
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Create datasets and dataloaders
    train_dataset = NoisyDataset(train_data, train_labels, lambda x: apply_markov_noise(x, 0.1, 10, noise_type))
    test_dataset = NoisyDataset(test_data, test_labels, lambda x: apply_markov_noise(x, 0.1, 10, noise_type))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Instantiate model, optimizer, and loss function
    model = DenoisingNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Train model
    epochs = 100
    for epoch in range(epochs):
        # Wrap train_loader with tqdm for a progress bar
        for noisy_data, clean_data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()
            outputs = model(noisy_data.float())
            loss = criterion(outputs, clean_data.float())
            loss.backward()
            optimizer.step()

    denoised_test_data = denoise_data(test_loader)

    # Select one image per label from test data
    unique_test_images, unique_test_labels = select_one_image_per_label(test_data, test_labels)

    # Generate noisy versions
    unique_noisy_images = apply_markov_noise(unique_test_images, 0.1, 10, noise_type)

    # Generate denoised versions
    unique_noisy_images_tensor = torch.tensor(unique_noisy_images, dtype=torch.float32)
    with torch.no_grad():
        unique_denoised_images_tensor = model(unique_noisy_images_tensor)
    unique_denoised_images = unique_denoised_images_tensor.numpy()

    # Visualize original, noisy, and denoised images
    plot_images(unique_test_images, unique_test_labels, num_images=10, title="Original Images")
    plot_images(unique_noisy_images, unique_test_labels, num_images=10, title="Noisy Images")
    plot_images(unique_denoised_images, unique_test_labels, num_images=10, title="Denoised Images")