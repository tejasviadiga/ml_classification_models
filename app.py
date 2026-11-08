import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

st.set_page_config(page_title="Breast Cancer Classifier Comparison", layout="wide")

MODEL_DIR = "model"
TARGET = "Diagnosis"
CLASS_NAMES = ["benign", "malignant"]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(MODEL_DIR, MODEL_FILES[name]))


def compute_metrics(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, prob),
        "Precision": precision_score(y, pred),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "MCC": matthews_corrcoef(y, pred),
    }


st.title("Breast Cancer Diagnosis: Classifier Comparison")
st.caption("Pre-trained models (UCI Breast Cancer Wisconsin Diagnostic). "
           "Upload a test CSV to evaluate them; nothing is retrained.")

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload test CSV", type="csv")
    model_name = st.selectbox("Select model", list(MODEL_FILES))
    st.markdown("The CSV must contain all 30 feature columns plus a `Diagnosis` "
                "column (0 = benign, 1 = malignant).")
    if os.path.exists("test_data.csv"):
        with open("test_data.csv", "rb") as f:
            st.download_button("Download sample test_data.csv", f, "test_data.csv", "text/csv")

if uploaded is None:
    st.info("Upload a test CSV in the sidebar to get started.")
    st.stop()

df = pd.read_csv(uploaded)
if TARGET not in df.columns:
    st.error(f"Column '{TARGET}' not found in the uploaded file.")
    st.stop()

model = load_model(model_name)
# Feature names/order the model was trained with (pipelines expose them via the first step)
expected = list(getattr(model, "feature_names_in_", []))
missing = [c for c in expected if c not in df.columns]
if missing:
    st.error(f"Missing feature columns: {missing}")
    st.stop()

X, y = df[expected], df[TARGET]
if y.nunique() < 2:
    st.error("The target column needs both classes present to compute AUC/MCC.")
    st.stop()

st.subheader("Data preview")
st.write(f"{len(df)} rows, {len(expected)} features")
st.dataframe(df.head(), width="stretch")

tab_single, tab_all = st.tabs([f"{model_name}: details", "All models comparison"])

with tab_single:
    metrics = compute_metrics(model, X, y)
    cols = st.columns(len(metrics))
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, f"{v:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(confusion_matrix(y, model.predict(X)), annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.markdown("**Classification report**")
        report = classification_report(y, model.predict(X), target_names=CLASS_NAMES,
                                       output_dict=True)
        st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

with tab_all:
    table = pd.DataFrame({n: compute_metrics(load_model(n), X, y) for n in MODEL_FILES}).T
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#cfe8cf"),
                 width="stretch")
    st.bar_chart(table)
    st.success(f"Best MCC on this data: **{table['MCC'].idxmax()}**")
