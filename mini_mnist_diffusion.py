import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import data
from tqdm import tqdm
import os
import argparse


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
            noisy_data += np.random.laplace(loc=0, scale=step_size, size=noisy_data.shape)
        elif noise_type == "uniform":
            noisy_data += np.random.uniform(low=-step_size, high=step_size, size=noisy_data.shape)
        elif noise_type == "poisson":
            noisy_data += np.random.poisson(lam=step_size, size=noisy_data.shape)
        elif noise_type == "binomial":
            noisy_data += np.random.binomial(n=1, p=0.5, size=noisy_data.shape) * step_size
        elif noise_type == "exponential":
            noisy_data += np.random.exponential(scale=step_size, size=noisy_data.shape)
        elif noise_type == "gamma":
            noisy_data += np.random.gamma(shape=2.0, scale=step_size, size=noisy_data.shape)
        elif noise_type == "beta":
            noisy_data += np.random.beta(a=0.5, b=0.5, size=noisy_data.shape) * step_size
        elif noise_type == "chisquare":
            noisy_data += np.random.chisquare(df=2, size=noisy_data.shape)
        elif noise_type == "rayleigh":
            noisy_data += np.random.rayleigh(scale=step_size, size=noisy_data.shape)
        elif noise_type == "logistic":
            noisy_data += np.random.logistic(loc=0, scale=step_size, size=noisy_data.shape)
        elif noise_type == "gumbel":
            noisy_data += np.random.gumbel(loc=0, scale=step_size, size=noisy_data.shape)
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
def denoise_data(loader, model, device):
    model.eval()
    denoised_data = []
    with torch.no_grad():
        for noisy_data, _ in tqdm(loader, desc="Denoising Test Data"):
            noisy_data = noisy_data.to(device)
            outputs = model(noisy_data.float())
            denoised_data.append(outputs.cpu().numpy())
    return np.concatenate(denoised_data, axis=0)


def select_one_image_per_label(data, labels):
    unique_images = []
    unique_labels = []
    for label in range(10):  # Loop through labels 0 to 9
        index = np.where(labels == label)[0][0]  # Find the first occurrence of each label
        unique_images.append(data[index])
        unique_labels.append(labels[index])
    return np.array(unique_images), np.array(unique_labels)


def plot_images(images, labels, noise_type, num_images=10, title="", save_directory="results/mini_mnist/"):
    # Create the directory if it does not exist
    save_directory += noise_type
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
    plt.close(fig)


if __name__ == '__main__':
    # Select type of noise and number of epochs
    parser = argparse.ArgumentParser(description="Denoising with PyTorch")
    parser.add_argument("--noise_type", type=str, default="normal", help="Type of noise to apply (default: normal)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")

    # Parse arguments and use them
    args = parser.parse_args()
    noise_type = args.noise_type
    epochs = args.epochs

    # Load the dataset
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Create datasets and dataloaders
    train_dataset = NoisyDataset(train_data, train_labels, lambda x: apply_markov_noise(x, 0.1, 10, noise_type))
    test_dataset = NoisyDataset(test_data, test_labels, lambda x: apply_markov_noise(x, 0.1, 10, noise_type))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Instantiate model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenoisingNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Train model
    for epoch in range(epochs):
        for noisy_data, clean_data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            noisy_data, clean_data = noisy_data.to(device), clean_data.to(device)
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
    plot_images(unique_test_images, unique_test_labels, noise_type, num_images=10, title="Original Images")
    plot_images(unique_noisy_images, unique_test_labels, noise_type, num_images=10, title="Noisy Images")
    plot_images(unique_denoised_images, unique_test_labels, noise_type, num_images=10, title="Denoised Images")