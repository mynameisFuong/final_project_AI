import joblib
import pandas as pd
import streamlit as st

from train_model import FEATURE_COLUMNS, MODEL_PATH, train_and_save_model


st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="centered",
)


def load_model_artifact():
    if not MODEL_PATH.exists():
        train_and_save_model()

    artifact = joblib.load(MODEL_PATH)
    if "models" not in artifact:
        train_and_save_model()
        artifact = joblib.load(MODEL_PATH)

    if "models" not in artifact:
        raise RuntimeError("Model artifact khong dung dinh dang. Hay chay lai: python train_model.py")

    return artifact


def predict_with_model(model, input_values: dict) -> tuple[int, float | None]:
    input_df = pd.DataFrame([input_values], columns=FEATURE_COLUMNS)
    prediction = int(model.predict(input_df)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_df)[0][1])

    return prediction, probability


def predict_all_models(input_values: dict) -> list[dict]:
    artifact = load_model_artifact()
    rows = []

    for model_name, model in artifact.get("models", {}).items():
        prediction, probability = predict_with_model(model, input_values)
        rows.append(
            {
                "Model": model_name,
                "Kết quả": "Có nguy cơ" if prediction == 1 else "Ít nguy cơ",
                "Xác suất nguy cơ": probability,
            }
        )

    return rows


def risk_label(probability: float | None, prediction: int) -> tuple[str, str]:
    if probability is None:
        return ("Nguy cơ cao", "danger") if prediction == 1 else ("Nguy cơ thấp", "success")

    if probability >= 0.65:
        return "Nguy cơ cao", "danger"
    if probability >= 0.4:
        return "Nguy cơ trung bình", "warning"
    return "Nguy cơ thấp", "success"


def build_metrics_table(metrics: dict) -> pd.DataFrame:
    rows = []
    for model_name, model_metrics in metrics.items():
        rows.append(
            {
                "Model": model_name,
                "Accuracy": model_metrics["accuracy"],
                "Precision": model_metrics["precision"],
                "Recall": model_metrics["recall"],
                "F1-score": model_metrics["f1"],
            }
        )
    return pd.DataFrame(rows)


artifact = load_model_artifact()
models = artifact.get("models")
if not models:
    st.error("Model artifact chua co du 3 model. Hay dung app va chay lai: python train_model.py")
    st.stop()

metrics = artifact.get("metrics", {})

st.title("Dự đoán nguy cơ tiểu đường")
st.caption("Demo AI so sánh Logistic Regression, Decision Tree và KNN.")

with st.sidebar:
    st.header("Thông tin model")
    selected_model_name = st.selectbox(
        "Chọn model để hiển thị kết quả chính",
        list(models.keys()),
        index=list(models.keys()).index("KNN") if "KNN" in models else 0,
    )
    st.write("Dataset: `dataset/diabetes_optimal.csv`")

    selected_metrics = metrics.get(selected_model_name)
    if selected_metrics:
        st.metric("Accuracy", f"{selected_metrics['accuracy'] * 100:.2f}%")
        st.metric("Precision", f"{selected_metrics['precision'] * 100:.2f}%")
        st.metric("Recall", f"{selected_metrics['recall'] * 100:.2f}%")
        st.metric("F1-score", f"{selected_metrics['f1'] * 100:.2f}%")

st.subheader("Nhập thông tin sức khỏe")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Số lần mang thai", min_value=0, max_value=20, value=1, step=1)
        glucose = st.number_input("Glucose", min_value=0.0, max_value=250.0, value=120.0, step=1.0)
        blood_pressure = st.number_input("Huyết áp", min_value=0.0, max_value=150.0, value=70.0, step=1.0)
        skin_thickness = st.number_input("Độ dày da", min_value=0.0, max_value=100.0, value=25.0, step=1.0)

    with col2:
        insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=120.0, step=1.0)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=30.0, step=0.1)
        pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.01,
        )
        age = st.number_input("Tuổi", min_value=1, max_value=120, value=35, step=1)

    submitted = st.form_submit_button("Dự đoán")

if submitted:
    values = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age,
    }

    selected_model = models[selected_model_name]
    prediction, probability = predict_with_model(selected_model, values)
    label, status = risk_label(probability, prediction)

    st.subheader(f"Kết quả chính: {selected_model_name}")

    if status == "danger":
        st.error(label)
    elif status == "warning":
        st.warning(label)
    else:
        st.success(label)

    if probability is not None:
        st.progress(probability)
        st.write(f"Xác suất mô hình dự đoán có nguy cơ: **{probability * 100:.2f}%**")

    st.write(
        "Kết quả phân loại:",
        "**Có nguy cơ tiểu đường**" if prediction == 1 else "**Ít nguy cơ tiểu đường**",
    )

    st.subheader("So sánh dự đoán của 3 model")
    predictions_df = pd.DataFrame(predict_all_models(values))
    st.dataframe(
        predictions_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Xác suất nguy cơ": st.column_config.ProgressColumn(
                "Xác suất nguy cơ",
                format="%.2f",
                min_value=0,
                max_value=1,
            )
        },
    )

    st.info(
        "Lưu ý: Đây là demo học máy phục vụ học tập, không thay thế kết luận hoặc tư vấn từ bác sĩ."
    )

st.subheader("Hiệu năng trên tập test")
if metrics:
    metrics_df = build_metrics_table(metrics)
    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Accuracy": st.column_config.ProgressColumn("Accuracy", format="%.2f", min_value=0, max_value=1),
            "Precision": st.column_config.ProgressColumn("Precision", format="%.2f", min_value=0, max_value=1),
            "Recall": st.column_config.ProgressColumn("Recall", format="%.2f", min_value=0, max_value=1),
            "F1-score": st.column_config.ProgressColumn("F1-score", format="%.2f", min_value=0, max_value=1),
        },
    )

with st.expander("Xem dữ liệu đầu vào mẫu"):
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Pregnancies": 6,
                    "Glucose": 148,
                    "BloodPressure": 72,
                    "SkinThickness": 35,
                    "Insulin": 169.5,
                    "BMI": 33.6,
                    "DiabetesPedigreeFunction": 0.627,
                    "Age": 50,
                }
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
