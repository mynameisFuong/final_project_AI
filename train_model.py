from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "diabetes_optimal.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "diabetes_models.joblib"

FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

MODEL_CONFIGS = {
    "Logistic Regression": Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=0.1,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    ),
    "Decision Tree": DecisionTreeClassifier(
        criterion="gini",
        max_depth=8,
        min_samples_leaf=2,
        min_samples_split=10,
        random_state=42,
    ),
    "KNN": Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                KNeighborsClassifier(
                    n_neighbors=17,
                    weights="uniform",
                    metric="manhattan",
                ),
            ),
        ]
    ),
}


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Khong tim thay dataset: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    missing_columns = set(FEATURE_COLUMNS + ["Outcome"]) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset thieu cot: {sorted(missing_columns)}")

    return data


def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }


def train_and_save_model() -> dict:
    data = load_dataset()
    x = data[FEATURE_COLUMNS]
    y = data["Outcome"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    trained_models = {}
    metrics = {}

    for model_name, model in MODEL_CONFIGS.items():
        model.fit(x_train, y_train)
        trained_models[model_name] = model
        metrics[model_name] = evaluate_model(model, x_test, y_test)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "models": trained_models,
            "features": FEATURE_COLUMNS,
            "metrics": metrics,
        },
        MODEL_PATH,
    )

    return metrics


if __name__ == "__main__":
    result = train_and_save_model()
    print(f"Da luu model vao: {MODEL_PATH}")
    for model_name, model_metrics in result.items():
        print(f"\n{model_name}")
        for metric_name, metric_value in model_metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
