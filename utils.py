import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from model import YourModelClass, Net  # Import Net if it's defined in model.py
import torch.optim as optim

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Net class if not imported
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Example for MNIST (28x28 images)
        self.fc2 = nn.Linear(128, 10)  # 10 output classes for MNIST

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc=f'Epoch={epoch} Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    epoch_loss = train_loss / len(train_loader)
    epoch_accuracy = 100 * correct / processed
    print(f'\nTrain Epoch: {epoch} \tLoss: {epoch_loss:.6f} \tAccuracy: {epoch_accuracy:.2f}%\n')
    
    return epoch_loss, epoch_accuracy

def evaluate_model(model, device, test_loader, criterion, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set after Epoch {epoch}: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    
    return test_loss, accuracy 

# Define your dataset class (example using MNIST)
class YourDatasetClass(Dataset):
    def __init__(self, transform=None):
        self.data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.transform:
            item = self.transform(item)
        return item

# Initialize the dataset and dataloader
transform = transforms.Compose([transforms.ToTensor()])  # Add any transformations you need
train_dataset = YourDatasetClass(transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Adjust batch_size as needed

# Initialize the test dataset and dataloader
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)  # Load test dataset
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # Adjust batch_size as needed

# Initialize the model
model = YourModelClass().to(device)

# Initialize the optimizer
#optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjust learning rate as needed

# Initialize the criterion
criterion = nn.CrossEntropyLoss()  # Use appropriate loss function for your task

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# Example training loop
num_epochs = 15  # Set the number of epochs
for epoch in range(1, num_epochs + 1):
    train_loss, train_accuracy = train(model, device, train_loader, optimizer, criterion, epoch)
    test_loss, test_accuracy = evaluate_model(model, device, test_loader, criterion, epoch)
