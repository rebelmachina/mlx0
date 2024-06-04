import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Step 1: Create synthetic dataset
def generate_data(num_samples):
    X = np.random.randint(1, 21, size=(num_samples, 2)).astype(np.float32)
    y = (X[:, 0] * X[:, 1]).astype(np.float32)
    return X, y

class CustomDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {'input': self.X[idx], 'output': self.y[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample

class LogTransform:
    def __call__(self, sample):
        input, output = sample['input'], sample['output']
        input = np.log1p(input)  # log1p to avoid log(0)
        output = np.log1p(output)
        return {'input': input, 'output': output}

# Generate data and split it
X, y = generate_data(1000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create datasets and loaders
train_dataset = CustomDataset(X_train, y_train, transform=LogTransform())
test_dataset = CustomDataset(X_test, y_test, transform=LogTransform())

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Step 2: Define the perceptron model
class Perceptron(nn.Module):
    def __init__(self, n_hidden_layers, n_hidden_neurons):
        super(Perceptron, self).__init__()
        self.hidden_layers = nn.ModuleList()

        self.hidden_layers.append(nn.Linear(2, n_hidden_neurons))

        for _ in range(n_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(n_hidden_neurons, n_hidden_neurons))
   
        self.output = nn.Linear(n_hidden_neurons, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        x = self.output(x)
        return x

model = Perceptron(4, 10)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)


# Step 3: Train the perceptron
num_epochs = 2000
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        inputs = batch['input']
        targets = batch['output'].view(-1, 1)
        inputs, targets = torch.tensor(inputs), torch.tensor(targets)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 4: Evaluate the perceptron
model.eval()
test_loss = 0
acc = 0
num_elements = 0
with torch.no_grad():
    for batch in test_loader:
        inputs = batch['input']
        targets = batch['output'].view(-1, 1)
        inputs, targets = inputs, targets

        predictions = model(inputs)
        batch_loss = criterion(predictions, targets)
        test_loss += batch_loss.item()

        acc += torch.sum(torch.isclose(predictions.view(-1), targets.view(-1), atol=1e-1))
        # print(f"predictions={predictions}, actual={targets}")


        num_elements += predictions.shape[0]

print(f'Test Loss: {test_loss / len(test_loader):.4f}')
print(f'Accuracy = {((acc / num_elements) * 100):.2f}%')


def multiply_numbers(model, a,b):

    with torch.no_grad():
        inp = torch.tensor([[a,b]], dtype=torch.float)
        inp = LogTransform()({"input": inp, "output": 0})
        out = model(inp['input'])

        return np.expm1(out).item()


a = 3
b = 21
print(f"{a} x {b} = {multiply_numbers(model, a,b):.2f}")
