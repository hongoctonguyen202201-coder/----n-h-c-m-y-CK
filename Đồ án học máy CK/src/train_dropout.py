
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader

# Import CNN đã xây dựng
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cnn_dropout import CNN

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

print("Thiet bi:", DEVICE)


# ==========================================
# 2. DOC DU LIEU
# ==========================================

data = np.load("outputs/nv2/shared_data.npz")

X_train = data["X_train"].reshape(-1, 1, 28, 28)
y_train = data["y_train"]

X_val = data["X_validation"].reshape(-1, 1, 28, 28)
y_val = data["y_validation"]

# Chuyen sang Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)


# ==========================================
# 3. TAO DATALOADER
# ==========================================

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

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
# 4. KHOI TAO CNN
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
# 5. HAM HUAN LUYEN
# ==========================================

history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

best_val_acc = 0.0

for epoch in range(EPOCHS):

    # ----- TRAIN -----
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total


    # ----- VALIDATION -----
    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss = val_running_loss / val_total
    val_acc = val_correct / val_total


    # Luu lich su
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)


    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_acc*100:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc*100:.2f}%")


    # Luu model tot nhat
    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "val_accuracy": best_val_acc,
                "epoch": epoch + 1
            },
                "models/cnn_dropout_best.pth"        )

        print("Da luu mo hinh tot nhat!")


# ==========================================
# 6. VE BIEU DO
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

plt.title("CNN - Loss")
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

plt.title("CNN - Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/cnn_dropout_history.png",
    dpi=150
)

plt.show()


print("\n================================")
print("HUAN LUYEN HOAN TAT!")
print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
print("Model: models/cnn_baseline_best.pth")
print("Chart: results/cnn_baseline_history.png")
print("================================")
