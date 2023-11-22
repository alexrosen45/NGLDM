import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import data
from tqdm import tqdm

# Load the dataset
train_data, train_labels, test_data, test_labels = data.load_all_data('data')

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
def apply_markov_noise(data, step_size=0.1, steps=10):
    noisy_data = data.copy()
    for _ in range(steps):
        noisy_data += np.random.normal(scale=step_size, size=noisy_data.shape)
    return noisy_data

# Create datasets and dataloaders
train_dataset = NoisyDataset(train_data, train_labels, lambda x: apply_markov_noise(x, 0.1, 10))
test_dataset = NoisyDataset(test_data, test_labels, lambda x: apply_markov_noise(x, 0.1, 10))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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

# Instantiate model, optimizer, and loss function
model = DenoisingNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train model
for epoch in range(100):
    # Wrap train_loader with tqdm for a progress bar
    for noisy_data, clean_data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{100}"):
        optimizer.zero_grad()
        outputs = model(noisy_data.float())
        loss = criterion(outputs, clean_data.float())
        loss.backward()
        optimizer.step()

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

denoised_test_data = denoise_data(test_loader)

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
plot_images(apply_markov_noise(test_data, 0.1, 10), test_labels, num_images=5, title="Noisy Images")
plot_images(denoised_test_data, test_labels, num_images=5, title="Denoised Images")
