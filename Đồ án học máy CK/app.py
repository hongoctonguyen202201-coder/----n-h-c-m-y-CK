# -*- coding: utf-8 -*-
"""
AI Handwriting Lab
Run from the project root:
    python -m streamlit run app.py

Expected project structure:
    app.py
    src/
    models/
    outputs/nv2/
    results/evaluation/
    data/digit-recognizer/train.csv
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

# Optional drawing component. The upload feature still works if it is unavailable.
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
    CANVAS_ERROR = ""
except Exception as exc:
    st_canvas = None
    CANVAS_AVAILABLE = False
    CANVAS_ERROR = str(exc)

try:
    import joblib
except Exception:
    joblib = None


# ---------------------------------------------------------------------
# 1. Paths and page configuration
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
NV2_DIR = OUTPUTS_DIR / "nv2"
RESULTS_DIR = ROOT / "results"
EVALUATION_DIR = RESULTS_DIR / "evaluation"
DATA_DIR = ROOT / "data" / "digit-recognizer"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

st.set_page_config(
    page_title="AI Handwriting Lab (Phòng thí nghiệm AI chữ viết tay)",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------
# 2. Styling
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #F5F7FB; color: #172033; }
    [data-testid="stHeader"] { background: rgba(245,247,251,.92); }
    .block-container { max-width: 1440px; padding-top: 1.5rem; padding-bottom: 3rem; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; color: #172033; }
    .hero {
        background: linear-gradient(120deg, #FFFFFF 0%, #EDF4FF 100%);
        border: 1px solid #DCE6F5; border-radius: 18px;
        padding: 24px 28px; margin-bottom: 18px;
    }
    .eyebrow { color: #2563EB; font-size: 11px; font-weight: 800; letter-spacing: 1.8px; }
    .hero-title { font-family: 'Manrope',sans-serif; font-size: 31px; font-weight: 800; margin: 4px 0; color: #14213D; }
    .hero-sub { color: #64748B; font-size: 13px; }
    .metric-card {
        background: #FFFFFF; border: 1px solid #E3EAF4; border-radius: 14px;
        padding: 16px 18px; min-height: 94px;
    }
    .metric-label { color: #64748B; font-size: 12px; }
    .metric-value { color: #172033; font-family: 'Manrope',sans-serif; font-size: 25px; font-weight: 800; margin-top: 4px; }
    .section-kicker { color: #2563EB; font-size: 11px; font-weight: 800; letter-spacing: 1.2px; }
    .soft-panel {
        background: #FFFFFF; border: 1px solid #E3EAF4; border-radius: 14px;
        padding: 16px; 
    }
    div[data-testid="stTabs"] button { font-weight: 700; }
    div.stButton > button, div.stDownloadButton > button {
        border-radius: 9px; font-weight: 700; min-height: 42px;
    }
    div.stButton > button[kind="primary"] {
        background: #2563EB; border-color: #2563EB;
    }
    [data-testid="stDataFrame"] { border: 1px solid #E3EAF4; border-radius: 10px; }
    .small-note { color: #64748B; font-size: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# 3. Helpers: data paths, image preprocessing, model loading
# ---------------------------------------------------------------------
CNN_CONFIG = {
    "CNN Baseline": ("cnn_model", "cnn_baseline_best.pth"),
    "CNN Dropout": ("cnn_dropout", "cnn_dropout_best.pth"),
    "CNN BatchNorm": ("cnn_batchnorm", "cnn_batchnorm_best.pth"),
    # Augmentation changes training images, not necessarily the network architecture.
    "CNN Augmentation": ("cnn_model", "cnn_augmentation_best.pth"),
    "CNN Combined": ("cnn_combined", "cnn_combined_best.pth"),
}

TRADITIONAL_ALIASES = {
    "KNN": ["knn"],
    "Logistic Regression": ["logistic_regression", "logistic", "lr"],
    "Random Forest": ["random_forest", "randomforest", "rf"],
    "SVM": ["svm_rbf", "svm_linear", "svm"],
    "LDA": ["lda", "linear_discriminant"],
}


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def find_files(folder: Path, suffixes: tuple[str, ...]) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in suffixes],
        key=lambda p: str(p).lower(),
    )


def prepare_digit_image(image: Image.Image | np.ndarray) -> np.ndarray:
    """
    Return a float32 28x28 array with dark background and bright digit,
    scaled to [0, 1]. This matches the project's inspected pixel range.
    """
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.ndim == 3:
            arr = np.asarray(Image.fromarray(arr.astype("uint8")).convert("L"))
        else:
            arr = arr.astype("float32")
            if arr.max(initial=0) <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype("uint8")
            arr = np.asarray(Image.fromarray(arr).convert("L"))
        pil = Image.fromarray(arr).convert("L")
    else:
        pil = image.convert("L")

    # Standardize polarity: digit should be bright on a dark background.
    a = np.asarray(pil, dtype=np.uint8)
    if float(a.mean()) > 127:
        pil = ImageOps.invert(pil)
        a = np.asarray(pil, dtype=np.uint8)

    # Remove faint background noise and locate the foreground.
    mask = a > max(18, int(a.max()) * 0.18)
    if mask.any():
        ys, xs = np.where(mask)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        cropped = pil.crop((x0, y0, x1, y1))
    else:
        cropped = pil

    # Fit digit into 20x20 while preserving aspect ratio; center on 28x28.
    w, h = cropped.size
    if w == 0 or h == 0:
        return np.zeros((28, 28), dtype=np.float32)
    scale = min(20.0 / w, 20.0 / h)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    resized = cropped.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    offset = ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2)
    canvas.paste(resized, offset)
    result = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.clip(result, 0.0, 1.0)


def resolve_state_dict(checkpoint: Any) -> dict:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint không chứa state_dict hợp lệ.")
    # Remove DataParallel prefix when present.
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}


@st.cache_resource(show_spinner="Đang nạp các mô hình CNN...")
def load_cnn_model(model_name: str):
    if model_name not in CNN_CONFIG:
        raise ValueError(f"Không có cấu hình cho {model_name}")
    module_name, checkpoint_name = CNN_CONFIG[model_name]
    checkpoint_path = MODELS_DIR / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

    module = importlib.import_module(module_name)
    model_class = getattr(module, "CNN", None)
    if model_class is None:
        raise AttributeError(f"{module_name}.py không có class CNN")

    model = model_class()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = resolve_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def predict_cnn(model_name: str, image_28: np.ndarray) -> tuple[int, np.ndarray]:
    model = load_cnn_model(model_name)
    tensor = torch.from_numpy(image_28).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return int(np.argmax(probabilities)), probabilities


def locate_traditional_artifact(model_name: str) -> Path | None:
    aliases = TRADITIONAL_ALIASES[model_name]
    candidates = find_files(NV2_DIR, (".joblib", ".pkl", ".pickle"))
    if not candidates:
        candidates = find_files(OUTPUTS_DIR, (".joblib", ".pkl", ".pickle"))

    # Prefer artifacts located in a matching model directory.
    matched = []
    for path in candidates:
        parent_text = " ".join(part.lower() for part in path.parts)
        if any(alias in parent_text for alias in aliases):
            matched.append(path)
    if model_name == "SVM":
        # Prefer RBF SVM when both linear and RBF artifacts exist.
        matched.sort(key=lambda p: (0 if "svm_rbf" in str(p).lower() else 1, str(p).lower()))
    return matched[0] if matched else None


@st.cache_resource(show_spinner="Đang nạp mô hình truyền thống...")
def load_traditional_model(model_name: str):
    if joblib is None:
        raise RuntimeError("Chưa cài joblib. Chạy: python -m pip install joblib")
    artifact = locate_traditional_artifact(model_name)
    if artifact is None:
        raise FileNotFoundError(
            f"Chưa tìm thấy file .joblib/.pkl của {model_name} trong outputs/nv2. "
            "Cần kiểm tra định dạng file mô hình đã lưu."
        )
    return joblib.load(artifact), artifact


def predict_traditional(model_name: str, image_28: np.ndarray) -> tuple[int, np.ndarray]:
    model, _ = load_traditional_model(model_name)
    x = image_28.reshape(1, -1).astype(np.float32)

    predicted = int(np.asarray(model.predict(x)).reshape(-1)[0])
    classes = getattr(model, "classes_", np.arange(10))
    probabilities = np.zeros(10, dtype=np.float64)

    if hasattr(model, "predict_proba"):
        raw = np.asarray(model.predict_proba(x))[0]
        for cls, prob in zip(classes, raw):
            if 0 <= int(cls) <= 9:
                probabilities[int(cls)] = float(prob)
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x))
        if scores.ndim == 1:
            # Binary decision output is not expected for 10 digit classes.
            scores = np.column_stack([-scores, scores])
        scores = scores.reshape(-1)
        scores = scores - np.max(scores)
        raw = np.exp(scores) / np.exp(scores).sum()
        for cls, prob in zip(classes, raw):
            if 0 <= int(cls) <= 9:
                probabilities[int(cls)] = float(prob)
    else:
        # No calibrated confidence available: show one-hot prediction honestly.
        probabilities[predicted] = 1.0

    total = probabilities.sum()
    if total > 0:
        probabilities /= total
    return predicted, probabilities


def model_status() -> pd.DataFrame:
    rows = []
    for name, (_, checkpoint) in CNN_CONFIG.items():
        path = MODELS_DIR / checkpoint
        rows.append({
            "Nhóm": "CNN",
            "Model (Mô hình)": name,
            "Trạng thái": "Ready (Sẵn sàng)" if path.exists() else "Missing checkpoint (Thiếu tệp trọng số)",
            "File (Tệp)": str(path.relative_to(ROOT)) if path.exists() else checkpoint,
        })
    for name in TRADITIONAL_ALIASES:
        artifact = locate_traditional_artifact(name)
        rows.append({
            "Nhóm": "Truyền thống",
            "Model (Mô hình)": name,
            "Trạng thái": "Ready (Sẵn sàng)" if artifact else "Model file not found (Chưa tìm thấy .pkl/.joblib)",
            "File (Tệp)": str(artifact.relative_to(ROOT)) if artifact else "Chưa xác định",
        })
    return pd.DataFrame(rows)


def find_evaluation_tables() -> list[Path]:
    return find_files(EVALUATION_DIR, (".csv", ".xlsx")) + find_files(NV2_DIR, (".csv", ".xlsx"))


def find_images() -> list[Path]:
    return find_files(EVALUATION_DIR, (".png", ".jpg", ".jpeg")) + find_files(RESULTS_DIR, (".png", ".jpg", ".jpeg"))


# ---------------------------------------------------------------------
# 4. Header
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">MACHINE LEARNING · COMPUTER VISION</div>
      <div class="hero-title">AI Handwriting Lab</div>
      <div class="hero-sub">Handwritten Digit Recognition & Model Evaluation · 0–9</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Main navigation
tab_recognition, tab_evaluation = st.tabs(
    ["01  ·  DIGIT RECOGNITION (NHẬN DIỆN CHỮ SỐ)", "02  ·  MODEL EVALUATION (ĐÁNH GIÁ MÔ HÌNH)"]
)


# ---------------------------------------------------------------------
# 5. Page 1: Recognition
# ---------------------------------------------------------------------
with tab_recognition:
    st.markdown('<div class="section-kicker">DIGIT RECOGNITION</div>', unsafe_allow_html=True)
    st.header("Digit Recognition (Nhận diện chữ số viết tay)")
    st.caption("Vẽ trực tiếp hoặc tải ảnh lên, sau đó chọn mô hình để thực hiện dự đoán.")

    model_options = list(CNN_CONFIG.keys()) + list(TRADITIONAL_ALIASES.keys())
    col_model, col_source = st.columns([1.1, 1.9])
    with col_model:
        selected_model = st.selectbox("Recognition Model (Mô hình nhận diện)", model_options, index=2)
        st.caption("CNN: 5 biến thể · Truyền thống: 5 thuật toán")
    with col_source:
        input_mode = st.radio(
            "Input Method (Phương thức nhập ảnh)",
            ["Draw Digit (Vẽ chữ số trực tiếp)", "Upload Image (Tải ảnh lên)"],
            horizontal=True,
        )

    image_array = None
    preview_image = None

    if input_mode == "Draw Digit (Vẽ chữ số trực tiếp)":
        draw_col, guide_col = st.columns([1.25, 1])
        with draw_col:
            st.markdown("**Canvas vẽ chữ số**")
            if CANVAS_AVAILABLE:
                canvas_result = st_canvas(
                    fill_color="rgba(0, 0, 0, 0)",
                    stroke_width=18,
                    stroke_color="#FFFFFF",
                    background_color="#000000",
                    height=280,
                    width=280,
                    drawing_mode="freedraw",
                    key="digit_canvas",
                    update_streamlit=True,
                )
                if canvas_result.image_data is not None:
                    rgba = canvas_result.image_data.astype(np.uint8)
                    if np.any(rgba[:, :, 3] > 0):
                        image_array = prepare_digit_image(rgba)
                        preview_image = Image.fromarray((image_array * 255).astype(np.uint8))
            else:
                st.error("Không thể nạp thư viện vẽ. Bạn vẫn có thể dùng chức năng tải ảnh.")
                st.code("python -m pip install streamlit-drawable-canvas==0.9.3")
        with guide_col:
            st.markdown("**Hướng dẫn**")
            st.markdown(
                """
                - Dùng chuột vẽ một chữ số từ 0 đến 9.
                - Vẽ chữ số lớn, nằm gần giữa vùng vẽ.
                - Nền canvas màu đen, nét vẽ màu trắng.
                - Chọn nút **Nhận diện** để chạy mô hình.
                """
            )
            st.info("Ảnh được chuẩn hóa về grayscale 28 × 28 và pixel trong khoảng [0, 1].")
    else:
        uploaded = st.file_uploader(
            "Chọn ảnh chữ số",
            type=["png", "jpg", "jpeg", "bmp"],
            accept_multiple_files=False,
        )
        if uploaded is not None:
            try:
                uploaded_image = Image.open(uploaded).convert("L")
                preview_image = uploaded_image
                image_array = prepare_digit_image(uploaded_image)
                st.image(uploaded_image, caption="Input Image (Ảnh đầu vào)", width=230)
            except Exception as exc:
                st.error(f"Không đọc được ảnh: {exc}")

    if preview_image is not None and input_mode == "Draw Digit (Vẽ chữ số trực tiếp)":
        st.image(preview_image, caption="Preprocessed Image (Ảnh sau tiền xử lý) 28 × 28", width=180)

    predict_col, clear_col = st.columns([1, 1])
    with predict_col:
        run_prediction = st.button("Recognize Digit (Nhận diện chữ số)", type="primary", use_container_width=True)
    with clear_col:
        if st.button("Clear Recognition History (Xóa lịch sử nhận diện)", use_container_width=True):
            st.session_state["prediction_history"] = []
            st.rerun()

    if "prediction_history" not in st.session_state:
        st.session_state["prediction_history"] = []

    if run_prediction:
        if image_array is None:
            st.warning("Bạn hãy vẽ một chữ số hoặc tải ảnh lên trước khi nhận diện.")
        else:
            try:
                with st.spinner(f"Đang dự đoán bằng {selected_model}..."):
                    if selected_model in CNN_CONFIG:
                        predicted_digit, probs = predict_cnn(selected_model, image_array)
                    else:
                        predicted_digit, probs = predict_traditional(selected_model, image_array)

                confidence = float(probs[predicted_digit])
                st.session_state["last_prediction"] = {
                    "model": selected_model,
                    "digit": predicted_digit,
                    "confidence (độ tin cậy)": confidence,
                    "probabilities": probs.tolist(),
                    "image": image_array.tolist(),
                }
                st.session_state["prediction_history"].insert(
                    0,
                    {
                        "Time (Thời điểm)": pd.Timestamp.now().strftime("%H:%M:%S %d/%m/%Y"),
                        "Model (Mô hình)": selected_model,
                        "Prediction (Dự đoán)": predicted_digit,
                        "Confidence (Độ tin cậy)": f"{confidence * 100:.2f}%",
                    },
                )
            except Exception as exc:
                st.error(f"Không thể dự đoán bằng {selected_model}.")
                st.exception(exc)

    result = st.session_state.get("last_prediction")
    if result:
        st.divider()
        st.markdown('<div class="section-kicker">PREDICTION RESULT</div>', unsafe_allow_html=True)
        result_col, detail_col = st.columns([0.8, 2.2])
        with result_col:
            st.markdown(
                f"""
                <div class="metric-card" style="text-align:center;padding:22px;">
                  <div class="metric-label">PREDICTED DIGIT</div>
                  <div style="font-family:Manrope;font-size:76px;font-weight:800;color:#2563EB;line-height:1.2;">
                    {result["digit"]}
                  </div>
                  <div style="font-weight:700;color:#15803D;">{result["confidence (độ tin cậy)"] * 100:.2f}% confidence</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(f"Mô hình: {result['model']}")
        with detail_col:
            st.markdown("**Phân phối xác suất theo từng chữ số**")
            probability_df = pd.DataFrame({
                "Digit (Chữ số)": [str(i) for i in range(10)],
                "Probability (Xác suất)": np.asarray(result["probabilities"]) * 100,
            }).set_index("Digit (Chữ số)")
            st.bar_chart(probability_df, y="Probability (Xác suất)", height=260)
            st.dataframe(
                probability_df.style.format({"Probability (Xác suất)": "{:.2f}%"}),
                use_container_width=True,
            )

    history = st.session_state.get("prediction_history", [])
    if history:
        st.divider()
        st.subheader("Recognition History (Lịch sử nhận diện)")
        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------
# 6. Page 2: Evaluation
# ---------------------------------------------------------------------
with tab_evaluation:
    st.markdown('<div class="section-kicker">MODEL EVALUATION</div>', unsafe_allow_html=True)
    st.header("Model Evaluation & Comparison (Đánh giá và so sánh mô hình)")
    st.caption("Trang này đọc các kết quả đã lưu trong dự án; không tự tạo hoặc giả lập chỉ số.")

    status_df = model_status()
    ready_count = int((status_df["Trạng thái"] == "Ready (Sẵn sàng)").sum())
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("TOTAL MODELS (TỔNG SỐ MÔ HÌNH)", "10")
    with c2:
        metric_card("MODEL FILES FOUND (TỆP MÔ HÌNH TÌM THẤY)", f"{ready_count}/10")
    with c3:
        metric_card("INFERENCE DEVICE (THIẾT BỊ DỰ ĐOÁN)", "CPU (Bộ xử lý trung tâm)")

    st.subheader("Model Status (Trạng thái mô hình)")
    st.dataframe(status_df, use_container_width=True, hide_index=True)

    if ready_count < 10:
        st.warning(
            "Một số mô hình chưa tìm thấy tệp trọng số hoặc artifact. "
            "Hãy kiểm tra định dạng file trong models/ và outputs/nv2/. "
            "Ứng dụng không tự giả định mô hình còn thiếu đã được huấn luyện."
        )

    st.divider()
    st.subheader("Saved Evaluation Results (Kết quả đánh giá đã lưu)")
    tables = find_evaluation_tables()
    if tables:
        table_labels = [str(p.relative_to(ROOT)) for p in tables]
        selected_table_label = st.selectbox("Select Results Table (Chọn bảng kết quả)", table_labels)
        selected_table = tables[table_labels.index(selected_table_label)]
        try:
            if selected_table.suffix.lower() == ".csv":
                evaluation_df = pd.read_csv(selected_table)
            else:
                evaluation_df = pd.read_excel(selected_table)
            st.dataframe(evaluation_df, use_container_width=True, hide_index=True)

            # Plot only when familiar metric columns are present.
            lower_to_original = {str(c).strip().lower().replace("-", "_").replace(" ", "_"): c
                                 for c in evaluation_df.columns}
            metric_candidates = ["accuracy", "precision", "recall", "f1", "f1_score"]
            found_metrics = [lower_to_original[k] for k in metric_candidates if k in lower_to_original]
            if found_metrics:
                numeric_df = evaluation_df[found_metrics].apply(pd.to_numeric, errors="coerce")
                if numeric_df.notna().any().any():
                    st.markdown("**Biểu đồ các chỉ số đánh giá**")
                    st.bar_chart(numeric_df)
        except Exception as exc:
            st.error(f"Không đọc được bảng kết quả: {exc}")
    else:
        st.info("Chưa tìm thấy CSV/XLSX kết quả trong results/evaluation hoặc outputs/nv2.")

    st.divider()
    st.subheader("Charts & Confusion Matrix (Biểu đồ và ma trận nhầm lẫn)")
    images = find_images()
    if images:
        image_labels = [str(p.relative_to(ROOT)) for p in images]
        selected_image_labels = st.multiselect(
            "Select Images to View (Chọn hình ảnh cần xem)",
            image_labels,
            default=image_labels[:4],
        )
        selected_paths = [images[image_labels.index(label)] for label in selected_image_labels]
        if selected_paths:
            cols = st.columns(2)
            for idx, path in enumerate(selected_paths):
                with cols[idx % 2]:
                    st.image(str(path), caption=path.name, use_container_width=True)
    else:
        st.info("Chưa tìm thấy hình ảnh đánh giá trong thư mục results/.")

    st.divider()
    st.subheader("Dataset Information (Thông tin dữ liệu)")
    train_csv = DATA_DIR / "train.csv"
    if train_csv.exists():
        try:
            data_info = pd.read_csv(train_csv, nrows=5)
            st.write(f"Đã tìm thấy dữ liệu huấn luyện: `{train_csv.relative_to(ROOT)}`")
            st.write("Các cột mẫu:")
            st.code(", ".join(map(str, data_info.columns[:12])))
            st.caption("Chỉ đọc 5 dòng đầu để kiểm tra cấu trúc, không nạp toàn bộ dữ liệu.")
        except Exception as exc:
            st.warning(f"Không đọc được file train.csv: {exc}")
    else:
        st.info("Không tìm thấy data/digit-recognizer/train.csv theo cấu trúc mặc định.")


# ---------------------------------------------------------------------
# 7. Footer
# ---------------------------------------------------------------------
st.markdown(
    """
    <br><hr style="border:0;border-top:1px solid #E2E8F0;">
    <div style="text-align:center;color:#94A3B8;font-size:11px;">
      AI HANDWRITING LAB · HANDWRITTEN DIGIT RECOGNITION · MACHINE LEARNING PROJECT
    </div>
    """,
    unsafe_allow_html=True,
)
