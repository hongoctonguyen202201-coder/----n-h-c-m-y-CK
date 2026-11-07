
import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)

        # Dropout: ngau nhien tat 30% neuron
        self.dropout = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)

        x = self.relu(self.fc1(x))

        # Ap dung Dropout khi huan luyen
        x = self.dropout(x)

        x = self.fc2(x)

        return x


if __name__ == "__main__":

    model = CNN()

    sample = torch.randn(8, 1, 28, 28)

    output = model(sample)

    print("===== CNN WITH DROPOUT =====")
    print("Input:", sample.shape)
    print("Output:", output.shape)
    print("Dropout:", model.dropout.p)
    print("Parameters:", sum(p.numel() for p in model.parameters()))

    print("\nMODEL CREATED SUCCESSFULLY!")
