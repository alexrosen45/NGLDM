import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load the CIFAR-10 dataset
transform = transforms.ToTensor()
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Define a PyTorch Dataset for CIFAR-10 with noise
class NoisyCIFARDataset(Dataset):
    def __init__(self, data, noise_fn):
        self.data = data
        self.noise_fn = noise_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, _ = self.data[idx]
        noisy_img = self.noise_fn(img)
        return noisy_img, img

# Apply Markov chain noise to the data (adapted for 3-channel images)
def apply_markov_noise(img, step_size=0.1, steps=10):
    noisy_img = img.clone()
    for _ in range(steps):
        noise = torch.randn_like(noisy_img) * step_size
        noisy_img += noise
        noisy_img = torch.clamp(noisy_img, 0, 1)  # Ensure pixel values are valid
    return noisy_img

# Create datasets and dataloaders
train_dataset = NoisyCIFARDataset(train_data, lambda x: apply_markov_noise(x, 0.1, 10))
test_dataset = NoisyCIFARDataset(test_data, lambda x: apply_markov_noise(x, 0.1, 10))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the CNN for denoising CIFAR-10 images
class DenoisingCNN(nn.Module):
    def __init__(self):
        super(DenoisingCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 3 input channels (RGB), 32 output channels
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 1024)  # Adjust the input features to match the flattened conv output
        self.fc2 = nn.Linear(1024, 3072)  # Output layer should match the input size (32*32*3)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # Flatten the tensor for the fully connected layer
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Using sigmoid to keep the output between 0 and 1
        return x.view(-1, 3, 32, 32)  # Reshape to match the size of CIFAR-10 images

# Instantiate model, optimizer, and loss function
model = DenoisingCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train model
epochs = 10
for epoch in range(epochs):
    for noisy_data, clean_data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
        optimizer.zero_grad()
        outputs = model(noisy_data)
        loss = criterion(outputs, clean_data)
        loss.backward()
        optimizer.step()

# Function to denoise data
def denoise_data(loader):
    model.eval()
    denoised_data = []
    with torch.no_grad():
        for noisy_data, _ in tqdm(loader, desc="Denoising Test Data"):
            outputs = model(noisy_data)
            denoised_data.append(outputs.cpu().numpy())
    return np.concatenate(denoised_data, axis=0)

# Denoise the test data
denoised_test_data = denoise_data(test_loader)

# Function to plot images
def plot_images(images, num_images=5, title=""):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i, ax in enumerate(axes):
        # Check if the image needs to be transposed
        if images[i].shape == (3, 32, 32):
            img = np.transpose(images[i], (1, 2, 0))  # Convert from PyTorch tensor to numpy and adjust channel order
        else:
            img = images[i]
        ax.imshow(img)
        ax.axis('off')
    plt.suptitle(title)
    plt.show()

# Visualize original, noisy, and denoised images
plot_images(test_data.data[:5], num_images=5, title="Original Images")
plot_images(apply_markov_noise(torch.stack([transform(img) for img in test_data.data[:5]]), 0.1, 10), num_images=5, title="Noisy Images")
plot_images(denoised_test_data[:5], num_images=5, title="Denoised Images")
