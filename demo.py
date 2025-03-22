import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys

# Check if GPU acceleration is enabled
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Fashion MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define a simple DNN model
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Instantiate model, loss, and optimizer
model = DNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    total_correct = 0
    total_samples = 0
    total_batches = len(train_dataloader)

    for batch_idx, (X_batch, y_batch) in enumerate(train_dataloader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == y_batch).sum().item()
        total_samples += y_batch.size(0)
        accuracy = total_correct / total_samples * 100
        progress = (batch_idx + 1) / total_batches * 100
        bar = '#' * int(progress // 2) + '-' * (50 - int(progress // 2))
        print(f"Epoch {epoch + 1}/{epochs} [{bar}] {progress:.1f}% - Acc: {accuracy:.2f}%, Loss: {loss.item():.4f}",
              end='\r')

    print(f"Epoch {epoch + 1}/{epochs} [{bar}] 100.0% - Final Acc: {accuracy:.2f}%, Loss: {loss.item():.4f} ")

print("Training completed!")