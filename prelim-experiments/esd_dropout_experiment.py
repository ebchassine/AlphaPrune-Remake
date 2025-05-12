
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.linalg import svdvals

# ----------------- DATA LOADING -----------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

# ----------------- MODEL DEFINITIONS -----------------
class OverfitCNN(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(OverfitCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(8 * 8 * 128, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32 -> 16
        x = self.pool(F.relu(self.conv2(x)))  # 16 -> 8
        x = x.view(-1, 8 * 8 * 128)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ----------------- TRAINING AND EVALUATION -----------------
def train(model, loader, optimizer, criterion):
    model.train()
    correct, total, loss_accum = 0, 0, 0.0
    for inputs, targets in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        loss_accum += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return loss_accum / len(loader), correct / total

def evaluate(model, loader, criterion):
    model.eval()
    correct, total, loss_accum = 0, 0, 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_accum += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return loss_accum / len(loader), correct / total

# ----------------- ESD AND DROPOUT ADJUSTMENT -----------------
def compute_esd_tail(weights, plot=False):
    singular_values = svdvals(weights)
    log_sv = np.log10(singular_values + 1e-10)
    hist, bin_edges = np.histogram(log_sv, bins=50, density=True)
    if plot:
        plt.plot(bin_edges[:-1], hist)
        plt.title("Log ESD of Weight Matrix")
        plt.show()
    tail_weight = np.mean(log_sv[log_sv > np.median(log_sv)])
    return min(max(tail_weight / 10, 0.1), 0.7)  # bound between [0.1, 0.7]

# ----------------- MAIN EXPERIMENT -----------------
def run_experiment():
    base_model = OverfitCNN()
    optimizer = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("Training overfit-prone model...")
    for epoch in range(10):
        train_loss, train_acc = train(base_model, train_loader, optimizer, criterion)
        print(f"Epoch {epoch + 1}: Train Acc = {train_acc:.4f}")

    test_loss, test_acc = evaluate(base_model, test_loader, criterion)
    print(f"Final Test Acc = {test_acc:.4f}")
    print(f"Generalization Gap: {train_acc - test_acc:.4f}")

    # Extract weights from fc1 layer and compute ESD tail
    weights = base_model.fc1.weight.detach().numpy()
    dropout_rate = compute_esd_tail(weights, plot=True)
    print(f"Calculated Dropout Rate based on ESD: {dropout_rate:.2f}")

    # Retrain with dropout
    dropout_model = OverfitCNN(dropout_rate)
    optimizer = torch.optim.Adam(dropout_model.parameters(), lr=1e-3)

    print("Retraining with dropout...")
    for epoch in range(10):
        train_loss, train_acc = train(dropout_model, train_loader, optimizer, criterion)
        print(f"Epoch {epoch + 1}: Train Acc = {train_acc:.4f}")

    test_loss, test_acc = evaluate(dropout_model, test_loader, criterion)
    print(f"Final Test Acc with Dropout = {test_acc:.4f}")
    print(f"Generalization Gap with Dropout: {train_acc - test_acc:.4f}")

if __name__ == "__main__":
    run_experiment()
