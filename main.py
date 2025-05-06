import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(  # Assume data coming from FashionMNIST
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for x, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {x.shape}")
    print(f"Shape of y {y.shape} {y.dtype}")
    break

device = torch.accelerator.current_accelerator(
).type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define model


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()  # Initialize pytorch components
        # Flattens [batch_size, 1, 28, 28] -> [batch_size, 784] since linear requires 2D input
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(  # Stack of layers
            nn.Linear(784, 512),
            nn.ReLU(),  # Takes max(0, x)
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):  # x is batch of images, gets flattened, passed through MLP (multilayer perceptron_ stack
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

# Optimize model parameters
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
# Single training loop, model makes predictions on trainint dataset and backpropagates predirection error to adjust parameters


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()  # enable training mode
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')


# Function to display multiple images with predicted and actual labels

def display_predictions(model, dataset, classes, num_images=6):
    model.eval()
    plt.figure(figsize=(12, 8))
    for i in range(num_images):
        image, label = dataset[i]
        with torch.no_grad():
            image_input = image.to(device).unsqueeze(0)  # add batch dimension
            output = model(image_input)
            predicted_label = output.argmax(1).item()

        plt.subplot(2, 3, i+1)
        plt.imshow(image.squeeze(), cmap="gray")
        plt.title(f'Pred: {classes[predicted_label]}\nTrue: {classes[label]}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# Call the display function
display_predictions(model, test_data, classes)
