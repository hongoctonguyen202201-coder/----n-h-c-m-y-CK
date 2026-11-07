
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 1. IMPORT CAC KIEN TRUC CNN
# ==========================================

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cnn_model import CNN as BaselineCNN
from cnn_dropout import CNN as DropoutCNN
from cnn_batchnorm import CNN as BatchNormCNN
from cnn_combined import CNN as CombinedCNN


# ==========================================
# 2. CAU HINH
# ==========================================

DEVICE = torch.device("cpu")
BATCH_SIZE = 128

os.makedirs("results/evaluation", exist_ok=True)

print("DEVICE:", DEVICE)


# ==========================================
# 3. DOC TAP TEST
# ==========================================

data = np.load("outputs/nv2/shared_data.npz")

X_test = data["X_test"].reshape(-1, 1, 28, 28)
y_test = data["y_test"]

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

test_dataset = TensorDataset(X_test, y_test)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\n===== TEST DATA =====")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ==========================================
# 4. DANH SACH MO HINH
# ==========================================

models = [
    (
        "Baseline",
        BaselineCNN,
        "models/cnn_baseline_best.pth"
    ),
    (
        "Dropout",
        DropoutCNN,
        "models/cnn_dropout_best.pth"
    ),
    (
        "BatchNorm",
        BatchNormCNN,
        "models/cnn_batchnorm_best.pth"
    ),
    (
        "Augmentation",
        BaselineCNN,
        "models/cnn_augmentation_best.pth"
    ),
    (
        "Combined",
        CombinedCNN,
        "models/cnn_combined_best.pth"
    )
]


# ==========================================
# 5. DANH GIA TUNG MO HINH
# ==========================================

results = []
all_confusion_matrices = []

for name, ModelClass, model_path in models:

    print("\n" + "=" * 45)
    print("EVALUATING:", name)

    if not os.path.exists(model_path):
        print("KHONG TIM THAY MODEL:", model_path)
        continue

    model = ModelClass().to(DEVICE)

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(predictions.cpu().numpy())


    # METRICS
    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true, y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(
        y_true, y_pred,
        labels=list(range(10))
    )

    all_confusion_matrices.append((name, cm))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })


    print(f"Accuracy : {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall   : {recall*100:.2f}%")
    print(f"F1-score : {f1*100:.2f}%")


    # CLASSIFICATION REPORT
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(10)),
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        f"results/evaluation/report_{name.lower()}.csv"
    )


    # CONFUSION MATRIX
    plt.figure(figsize=(7, 6))

    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix - {name}")
    plt.colorbar()

    plt.xticks(range(10))
    plt.yticks(range(10))

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for i in range(10):
        for j in range(10):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=8
            )

    plt.tight_layout()

    plt.savefig(
        f"results/evaluation/cm_{name.lower()}.png",
        dpi=150
    )

    plt.close()

    print("Evaluation completed!")


# ==========================================
# 6. TONG HOP KET QUA
# ==========================================

if len(results) == 0:
    raise RuntimeError("Khong co mo hinh nao duoc danh gia!")

df_results = pd.DataFrame(results)

df_results = df_results.sort_values(
    by="Accuracy",
    ascending=False
)

df_results.to_csv(
    "results/evaluation/model_comparison.csv",
    index=False
)

print("\n========================================")
print("FINAL MODEL COMPARISON")
print("========================================")

print(
    df_results.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4%}".format,
            "Precision": "{:.4%}".format,
            "Recall": "{:.4%}".format,
            "F1-score": "{:.4%}".format
        }
    )
)

print("\nBEST MODEL:")
print(df_results.iloc[0]["Model"])

print("\nRESULTS SAVED IN results/evaluation/")
print("EVALUATION COMPLETED!")
