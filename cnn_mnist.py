import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import itertools
from tqdm import tqdm


class SimpleCNN(nn.Module):
    def __init__(self, kernel_sizes, channel_sizes, fc_size, num_classes):
        super(SimpleCNN, self).__init__()
        self.layers = nn.ModuleList()
        in_channels = 1

        for kernel_size, out_channels in zip(kernel_sizes, channel_sizes):
            self.layers.append(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, padding=kernel_size // 2
                )
            )
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(2))
            self.layers.append(nn.Dropout(0.25))
            in_channels = out_channels

        # Calculate the size of the flattened feature map
        self.feature_size = in_channels * (28 // (2 ** len(kernel_sizes))) ** 2

        self.fc = nn.Linear(self.feature_size, fc_size)
        self.output = nn.Linear(fc_size, num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        x = self.output(x)
        return x


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    pbar = tqdm(test_loader, desc="Evaluating")
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            pbar.set_postfix({"accuracy": f"{correct/total:.4f}"})
    return correct / total


def hyperparameter_search():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    num_classes = 10
    num_epochs = 10

    # Define the hyperparameter search space
    kernel_sizes_options = [[3], [3, 3], [3, 3, 3], [5], [5, 5], [5, 5, 5]]
    channel_sizes_options = [
        [32],
        [32, 64],
        [32, 64, 128],
        [16],
        [16, 32],
        [16, 32, 64],
    ]
    fc_sizes = [64, 128, 256]

    best_accuracy = 0
    best_model = None
    best_params = None

    combinations = list(
        itertools.product(kernel_sizes_options, channel_sizes_options, fc_sizes)
    )
    pbar = tqdm(combinations, desc="Hyperparameter Search")

    for kernel_sizes, channel_sizes, fc_size in pbar:
        if len(kernel_sizes) != len(channel_sizes):
            continue  # Skip if the number of layers doesn't match

        model = SimpleCNN(kernel_sizes, channel_sizes, fc_size, num_classes).to(device)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            train(model, train_loader, optimizer, criterion, device)

        accuracy = evaluate(model, test_loader, device)

        pbar.set_postfix(
            {
                "Kernels": str(kernel_sizes),
                "Channels": str(channel_sizes),
                "FC": fc_size,
                "Accuracy": f"{accuracy:.4f}",
            }
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_params = (kernel_sizes, channel_sizes, fc_size)

    print(
        f"\nBest model - Kernel sizes: {best_params[0]}, Channel sizes: {best_params[1]}, FC size: {best_params[2]}, Accuracy: {best_accuracy:.4f}"
    )
    return best_model, best_params, best_accuracy


if __name__ == "__main__":
    best_model, best_params, best_accuracy = hyperparameter_search()
