import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 1
batch_size = 64
learning_rate = 0.001

# CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.batchnorm(self.conv1(x)))
        x = self.relu(self.batchnorm(self.conv2(x)))
        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Downsampling
        self.down1 = ConvBlock(3, 64)
        self.down2 = ConvBlock(64, 128)
        self.down3 = ConvBlock(128, 256)
        self.down4 = ConvBlock(256, 512)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Upsampling
        self.up4 = ConvBlock(1024 + 512, 512)
        self.up3 = ConvBlock(512 + 256, 256)
        self.up2 = ConvBlock(256 + 128, 128)
        self.up1 = ConvBlock(128 + 64, 64)

        # Final Convolution
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        # Downsampling
        x1 = self.down1(x)
        x2 = self.down2(F.max_pool2d(x1, 2))
        x3 = self.down3(F.max_pool2d(x2, 2))
        x4 = self.down4(F.max_pool2d(x3, 2))

        # Bottleneck
        x5 = self.bottleneck(F.max_pool2d(x4, 2))

        # Upsampling + Concatenation
        x = F.interpolate(x5, scale_factor=2)
        x = torch.cat([x, x4], dim=1)
        x = self.up4(x)

        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, x3], dim=1)
        x = self.up3(x)

        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x)

        x = F.interpolate(x, scale_factor=2)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x)

        # Final Convolution
        x = self.final_conv(x)
        return x

model = UNet().to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Function to add noise
def add_noise(inputs):
    noise = torch.randn_like(inputs) * 0.1  # Adjust noise level
    return inputs + noise, noise

for epoch in range(num_epochs):
    train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)

    for images, _ in train_loader_tqdm:
        images = images.to(device)
        
        # Add noise
        noisy_images, noise = add_noise(images)

        # Forward pass
        outputs = model(noisy_images)
        loss = criterion(outputs, noise)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the progress bar
        train_loader_tqdm.set_postfix(loss=loss.item())

    train_loader_tqdm.close()


# Test the model
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)


def save_model_checkpoint(model, name, save_directory="models/"):
    # Create the directory if it does not exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Save the model checkpoint
    torch.save(model.state_dict(), save_directory+name+".ckpt")


save_model_checkpoint(model, "unet_diffusion_model")
