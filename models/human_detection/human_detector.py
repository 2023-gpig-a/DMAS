import pathlib
import sys
import torch
import torchvision.datasets as datasets
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle

from torch.utils.data import DataLoader, random_split
from torchvision import transforms

'''
Human Detection Dataset
Not sure if this is what we will use in the end, it only has a few hundred images, but it was helpful for testing
Available at: https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset
Acquired on Wednesday 29th November
'''


class Classifier(nn.Module):

    # Create the train and val dataloaders and the model
    def __init__(self, batch_size: int = 32, pretrained: bool = True):
        super(Classifier, self).__init__()

        # Create dataloaders
        self.batch_size = batch_size
        self.classes = ["No People", "Yes People"]
        self.tensor_transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Resize((128, 128), antialias=True),
            transforms.Lambda(lambda y: y.float() / 255),
            transforms.Normalize(0.5, 0.5),
        ])

        # Define Model
        self.conv_layers = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # B x 4 x 64 x 64

            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # B x 16 x 32 x 32

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # B x 32 x 16 x 16

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # B x 16 x 8 x 8

            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # B x 8 x 4 x 4

        )
        self.MLP = nn.Sequential(
            nn.Linear(in_features=8 * 4 * 4, out_features=4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Linear(in_features=4, out_features=2),
            nn.Softmax(dim=1)
        )

        total_parameters = (sum(p.numel() for p in self.conv_layers.parameters() if p.requires_grad) +
                            sum(p.numel() for p in self.MLP.parameters() if p.requires_grad))
        print(f"Created classifier with {total_parameters} trainable parameters")

        if pretrained:
            self.load()

    def forward(self, x):
        # Input x has dimensions B x 3 x 128 x 128, B is batch size
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.MLP(x)
        return x  # Output has dimensions B x 2

    def load(self, state_dict_loc: str = None) -> None:
        if state_dict_loc is None:
            state_dict_loc = "models/human_detection/weights/human_classification_weights.pkl"
        self.load_state_dict(torch.load(state_dict_loc))
        self.eval()


def train(model: nn.Module, epochs: int = 0, device: str = "cpu", results_out: str = None, weights_out: str = None):

    # Set up our training environment
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters())

    dataset_src = "datasets/human detection dataset"
    try:
        people_dataset = datasets.ImageFolder(root=dataset_src, transform=model.tensor_transform)
    except FileNotFoundError:
        print('Dataset not found. Make sure you have downloaded the human detection dataset from')
        print('https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset')
        print(f'and saved it to {dataset_src}')
        sys.exit(1)
    train_size = int(0.8 * len(people_dataset))
    test_size = len(people_dataset) - train_size
    train_dataset, validation_dataset = random_split(people_dataset,[train_size, test_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=model.batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=model.batch_size, shuffle=True)

    loss_func = nn.CrossEntropyLoss()

    # These variables will store the data for analysis
    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []

    # Training
    for epoch in range(epochs):

        # Train the model and evaluate on the training set
        total_loss = 0
        correct = 0
        total = 0
        total_loss = 0

        for i, (inputs, labels) in enumerate(train_loader):

            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            loss = loss_func(output, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            y_pred = torch.argmax(output, 1)
            correct += (y_pred == labels).sum()
            total += float(labels.size(0))
            total_loss += loss * inputs.shape[0]

        total_loss /= len(train_dataset)
        training_losses.append(total_loss.item())
        training_accuracies.append((correct / total).item())
        train_accuracy = (correct / total).item()

        # Evaluate the model on the validation set
        # Reset counters and switch to eval mode
        correct = 0
        total = 0
        model.eval()

        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                output = model(inputs)
                loss = loss_func(output, labels)
                y_pred = torch.argmax(output, 1)
                correct += (y_pred == labels).sum()
                total += float(labels.size(0))
                total_loss += loss * inputs.shape[0]
            validation_accuracy = (correct / total).item()
        total_loss /= len(validation_dataset)

        # Switch back to train mode and save counters
        model.train()
        validation_accuracies.append(validation_accuracy)
        validation_losses.append(total_loss.item())
        print(
            f'Epoch {epoch + 1}, '
            f'Train Accuracy: {round(train_accuracy,2)}, '
            f'Validation Accuracy: {round(validation_accuracy,2)}'
        )

        if results_out is not None:
            data = {
                "train_accuracy": training_accuracies,
                "train_losses": training_losses,
                "val_accuracy": validation_accuracies,
                "val_losses": validation_losses,
            }
            # Ensure the weights directory exists
            pathlib.Path(results_out).parent.mkdir(parents=True, exist_ok=True)
            with open(results_out, "wb+") as file:
                pickle.dump(data, file)

        if weights_out is not None:
            torch.save(model.state_dict(), weights_out)


def display_training_graphs(src_location: str):
    with open(src_location, "rb") as file:
        data = pickle.load(file)

    plt.title("Training curve")
    plt.plot(range(len(data["train_losses"])), data["train_losses"], 'r', label='Train')
    plt.plot(range(len(data["val_losses"])), data["val_losses"], 'g', label='Val')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="lower right")
    plt.show()

    plt.title("Classification accuracy")
    plt.plot(range(len(data["train_accuracy"])), data["train_accuracy"], 'r', label='Train')
    plt.plot(range(len(data["val_accuracy"])), data["val_accuracy"], 'g', label='Val')
    plt.xlabel("Epoch")
    plt.ylabel("Classification accuracy")
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    # Run this to train the model and save the weights
    model = Classifier(batch_size=6, pretrained=False)

    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    if torch.cuda.is_available():
        device = "cuda"

    train(
        model,
        20,
        device,
        "weights/human_classification_results.pkl",
        "weights/human_classification_weights.pkl"
    )
    display_training_graphs("human_classification_results.pkl")
