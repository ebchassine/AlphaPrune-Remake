import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.linalg import svdvals
import os


# ----------------- DATA LOADING -----------------
def load_cifar10(data_dir):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    return train_set, test_set


# ----------------- MODEL DEFINITIONS -----------------
class RobustCNN(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(RobustCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(dropout_rate),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * 256, 512), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        return x


# ----------------- TRAINING AND EVALUATION -----------------
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += outputs.argmax(1).eq(y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * y.size(0)
            correct += outputs.argmax(1).eq(y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


# ----------------- ESD AND DROPOUT ADJUSTMENT -----------------
def compute_esd_tail(weights, plot_path):
    singular_values = svdvals(weights)
    log_sv = np.log10(singular_values + 1e-10)
    plt.figure()
    plt.hist(log_sv, bins=40, density=True)
    plt.title("Histogram of Log ESD (fc1 weights)")
    plt.xlabel("log10(Singular Value)")
    plt.ylabel("Density")
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()
    tail_weight = np.mean(log_sv[log_sv > np.median(log_sv)])
    return min(max(tail_weight / 10, 0.1), 0.7)


# ----------------- PLOTTING -----------------
def plot_loss(train_losses, val_losses, out_path):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()


# ----------------- MAIN PIPELINE -----------------
def run():
    os.makedirs("artifacts", exist_ok=True)
    data_dir = "./data"
    train_set, test_set = load_cifar10(data_dir)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    # ----- First model (no dropout) -----
    print("Training initial model (no dropout)...")
    model = RobustCNN(dropout_rate=0.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_losses, val_losses = [], []

    for epoch in range(15):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"[Epoch {epoch+1}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), "artifacts/model_nodropout.pt")
    plot_loss(train_losses, val_losses, "artifacts/loss_curve_nodropout.png")

    # ----- ESD Analysis -----
    weights = model.fc_block[1].weight.detach().cpu().numpy()
    dropout_rate = compute_esd_tail(weights, "artifacts/esd_histogram.png")
    print(f"Calculated dropout rate from ESD: {dropout_rate:.2f}")

    # ----- Second model (ESD-based dropout) -----
    print("Training new model with ESD-informed dropout...")
    model_do = RobustCNN(dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.Adam(model_do.parameters(), lr=1e-3)
    train_losses_do, val_losses_do = [], []

    for epoch in range(15):
        train_loss, train_acc = train(model_do, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model_do, test_loader, criterion, device)
        train_losses_do.append(train_loss)
        val_losses_do.append(val_loss)
        print(f"[Epoch {epoch+1}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    torch.save(model_do.state_dict(), "artifacts/model_dropout.pt")
    plot_loss(train_losses_do, val_losses_do, "artifacts/loss_curve_dropout.png")


if __name__ == "__main__":
    run()
