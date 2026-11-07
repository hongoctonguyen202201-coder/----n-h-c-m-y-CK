
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Đọc dữ liệu
df = pd.read_csv("data/digit-recognizer/train.csv")

# 2. Lấy nhãn chữ số
labels = df["label"].values

# 3. Lấy 784 pixel
images = df.drop(columns=["label"]).values

# 4. Chọn 10 ảnh đầu tiên
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, ax in enumerate(axes.flat):
    # Chuyển 784 pixel thành ảnh 28x28
    image = images[i].reshape(28, 28)

    # Hiển thị ảnh
    ax.imshow(image, cmap="gray")

    # Hiển thị nhãn thật
    ax.set_title(f"Label: {labels[i]}")

    ax.axis("off")

plt.suptitle("Handwritten Digit Recognition - Sample Images")
plt.tight_layout()

# Lưu hình ảnh
plt.savefig("outputs/sample_digits.png", dpi=150)

# Hiển thị
plt.show()
