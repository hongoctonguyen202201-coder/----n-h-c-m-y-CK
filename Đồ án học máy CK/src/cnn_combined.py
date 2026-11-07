
import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        # Convolution block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Convolution block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten
        x = torch.flatten(x, start_dim=1)

        # Fully connected
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


if __name__ == "__main__":

    model = CNN()
    sample = torch.randn(8, 1, 28, 28)

    output = model(sample)

    print("===== CNN COMBINED =====")
    print("Input:", sample.shape)
    print("Output:", output.shape)
    print("Parameters:", sum(
        p.numel() for p in model.parameters()
    ))

    print("CNN COMBINED CREATED SUCCESSFULLY!")
