
import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # Convolution layer 1
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # Convolution layer 2
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        # Max Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

    def forward(self, x):

        # Block 1
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        # Block 2
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten
        x = torch.flatten(x, start_dim=1)

        # Dense layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# Kiểm tra mô hình
if __name__ == "__main__":

    model = CNN()

    # Tạo thử 1 batch gồm 8 ảnh
    sample = torch.randn(8, 1, 28, 28)

    output = model(sample)

    print("===== KIEM TRA CNN =====")
    print("Input shape:", sample.shape)
    print("Output shape:", output.shape)
    print("So tham so:", sum(p.numel() for p in model.parameters()))

    print("\nCNN HOAT DONG THANH CONG!")
