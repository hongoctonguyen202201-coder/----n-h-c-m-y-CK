
import numpy as np
import torch

# =====================================
# 1. ĐỌC DỮ LIỆU DÙNG CHUNG CỦA NHÓM
# =====================================

data = np.load("outputs/nv2/shared_data.npz")

X_train = data["X_train"]
y_train = data["y_train"]

X_val = data["X_validation"]
y_val = data["y_validation"]

X_test = data["X_test"]
y_test = data["y_test"]

# =====================================
# 2. KIỂM TRA DỮ LIỆU
# =====================================

print("===== THONG TIN DU LIEU =====")

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

print("\nPixel min:", X_train.min())
print("Pixel max:", X_train.max())

print("\nCac nhan:", np.unique(y_train))

# =====================================
# 3. CHUYỂN ĐỔI SANG DẠNG CNN
# =====================================

X_train = X_train.reshape(-1, 1, 28, 28)
X_val = X_val.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)

# =====================================
# 4. CHUYỂN SANG PYTORCH TENSOR
# =====================================

X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train).long()

X_val = torch.from_numpy(X_val)
y_val = torch.from_numpy(y_val).long()

X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test).long()

# =====================================
# 5. KIỂM TRA KẾT QUẢ
# =====================================

print("\n===== SAU KHI CHUYEN DOI =====")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("X_test:", X_test.shape)

print("\nKieu du lieu:", X_train.dtype)
print("PyTorch:", torch.__version__)

print("\nHOAN THANH KIEM TRA!")
