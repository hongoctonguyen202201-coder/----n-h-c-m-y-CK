
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Import CNN Baseline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cnn_model import CNN


# ==========================================
# 1. CAU HINH
# ==========================================

SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cpu")

print("Device:", DEVICE)


# ==========================================
# 2. DATA AUGMENTATION
# ==========================================

augmentation = transforms.RandomAffine(
    degrees=10,
    translate=(0.08, 0.08),
    fill=0
)

print("Data Augmentation:")
print("- Rotation: +/- 10 degrees")
print("- Translation: 8%")


# ==========================================
# 3. DATASET
# ==========================================

class DigitDataset(Dataset):

    def __init__(self, images, labels, transform=None):
        self.images = torch.tensor(
            images.reshape(-1, 1, 28, 28),
            dtype=torch.float32
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.long
        )

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        image = self.images[index]
        label = self.labels[index]

        # Chi tang cuong anh Train
        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ==========================================
# 4. DOC DU LIEU
# ==========================================

data = np.load("outputs/nv2/shared_data.npz")

X_train = data["X_train"]
y_train = data["y_train"]

X_val = data["X_validation"]
y_val = data["y_validation"]

train_dataset = DigitDataset(
    X_train,
    y_train,
    transform=augmentation
)

# Validation khong tang cuong
val_dataset = DigitDataset(
    X_val,
    y_val,
    transform=None
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==========================================
# 5. KHOI TAO MO HINH
# ==========================================

model = CNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ==========================================
# 6. HUAN LUYEN
# ==========================================

history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

best_val_acc = 0.0

for epoch in range(EPOCHS):

    model.train()

    train_loss_total = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss_total += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)

    train_loss = train_loss_total / train_total
    train_acc = train_correct / train_total


    # VALIDATION
    model.eval()

    val_loss_total = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss_total += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss = val_loss_total / val_total
    val_acc = val_correct / val_total


    # LUU LICH SU
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)


    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_acc*100:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc*100:.2f}%")


    # LUU CHECKPOINT TOT NHAT
    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "val_accuracy": best_val_acc,
                "epoch": epoch + 1
            },
            "models/cnn_augmentation_best.pth"
        )

        print("Da luu mo hinh tot nhat!")


# ==========================================
# 7. VE BIEU DO
# ==========================================

epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(
    epochs_range,
    history["train_loss"],
    marker="o",
    label="Train Loss"
)

plt.plot(
    epochs_range,
    history["val_loss"],
    marker="o",
    label="Validation Loss"
)

plt.title("CNN + Data Augmentation - Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)


plt.subplot(1, 2, 2)

plt.plot(
    epochs_range,
    history["train_acc"],
    marker="o",
    label="Train Accuracy"
)

plt.plot(
    epochs_range,
    history["val_acc"],
    marker="o",
    label="Validation Accuracy"
)

plt.title("CNN + Data Augmentation - Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/cnn_augmentation_history.png",
    dpi=150
)

plt.show()


print("\n================================")
print("TRAINING COMPLETED!")
print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
print("Model: models/cnn_augmentation_best.pth")
print("Chart: results/cnn_augmentation_history.png")
print("================================")
