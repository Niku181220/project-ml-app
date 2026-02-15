# app.py
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from joblib import load
from sklearn.metrics import confusion_matrix, classification_report

ARTIFACT_DIR = os.path.join("model", "artifacts")
SUMMARY_PATH = os.path.join(ARTIFACT_DIR, "metrics_summary.json")

st.set_page_config(page_title="ML Assignment 2 - Classification Models", layout="wide")


@st.cache_data
def load_summary():
    if not os.path.exists(SUMMARY_PATH):
        return None
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_model(model_file: str):
    return load(model_file)


def plot_confusion_matrix(cm, class_names):
    fig = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=30, ha="right")
    plt.yticks(ticks=np.arange(len(class_names)), labels=class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    return fig


st.title("Classification Models (Streamlit App)")

summary = load_summary()
if summary is None:
    st.error("Model artifacts not found. First run training:\n\n"
             "`python model/train_and_save.py --data data/dataset.csv --target <TARGET_COLUMN>`")
    st.stop()

st.caption(f"Dataset: {summary['dataset_path']} | Target: {summary['target_column']} | Task: {summary['task_type']}")

# Left panel: model selection
model_names = [m["name"] for m in summary["models"]]
selected_model = st.selectbox("Select a model", model_names)

# Load selected model pipeline
selected_meta = next(m for m in summary["models"] if m["name"] == selected_model)
model_path = selected_meta["file"]
pipe = load_model(model_path)

# Display metrics table (from held-out split)
st.subheader("Evaluation Metrics (from Train/Test split during training)")
table_df = pd.DataFrame(summary["table"])
st.dataframe(table_df, use_container_width=True)

# Display selected model details
details = summary["details"][selected_model]
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Accuracy", f"{details['Accuracy']:.4f}")
c2.metric("AUC", f"{details['AUC']:.4f}")
c3.metric("Precision", f"{details['Precision']:.4f}")
c4.metric("Recall", f"{details['Recall']:.4f}")
c5.metric("F1", f"{details['F1']:.4f}")
c6.metric("MCC", f"{details['MCC']:.4f}")

st.divider()

# Confusion matrix + classification report (saved from training test split)
st.subheader("Confusion Matrix / Classification Report (Training Test Split)")
label_map = summary.get("label_map", {})
labels = summary.get("labels", [])
class_names = [label_map.get(int(x), str(x)) for x in labels]

cm_saved = np.array(details["ConfusionMatrix"])
fig = plot_confusion_matrix(cm_saved, class_names)
st.pyplot(fig)

st.text("Classification Report:")
st.code(details["ClassificationReport"])

st.divider()

# Upload test CSV
st.subheader("Upload CSV (Test Data)")
st.write("Upload **only test data** (recommended). If your CSV also contains the target column, the app will compute metrics on the uploaded file too.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    df_up = pd.read_csv(uploaded)
    st.write("Preview:")
    st.dataframe(df_up.head(10), use_container_width=True)

    target_col = summary["target_column"]
    has_target = target_col in df_up.columns

    # Build X
    if has_target:
        X_up = df_up.drop(columns=[target_col])
    else:
        X_up = df_up.copy()

    # Predict
    preds_num = pipe.predict(X_up)

    # Show readable prediction labels if mapping exists
    pred_labels = [label_map.get(int(p), str(p)) for p in preds_num]
    pred_df = pd.DataFrame({"prediction": pred_labels})
    st.write("Predictions:")
    st.dataframe(pred_df.head(50), use_container_width=True)

    # Probabilities (optional)
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        proba = pipe.predict_proba(X_up)
        proba_df = pred_df.copy()
        if summary["task_type"] == "binary":
            proba_df["prob_class_1"] = proba[:, 1]
        else:
            for i, lab in enumerate(labels):
                proba_df[f"prob_{label_map.get(int(lab), str(lab))}"] = proba[:, i]
        st.write("Predictions + Probabilities:")
        st.dataframe(proba_df.head(50), use_container_width=True)

    #Uploaded test metrics ONLY if target column exists AND can be encoded safely
    if has_target:
        st.subheader("Uploaded Test Metrics (only if target column exists in uploaded CSV)")

        # y_true from uploaded file
        y_raw = df_up[target_col]

        # Convert y_true to numeric labels used in training
        y_up = None

        # If already numeric, try directly
        if np.issubdtype(y_raw.dtype, np.number):
            y_up = y_raw.astype(int)
        else:
            # Map string labels (e.g., 'B','M') -> 0/1 using inverse label_map
            inv_label_map = {str(v): int(k) for k, v in label_map.items()}
            y_mapped = y_raw.astype(str).map(inv_label_map)

            if y_mapped.isna().any():
                st.warning(
                    f"Target column '{target_col}' values don't match training labels.\n\n"
                    f"Expected one of: {list(inv_label_map.keys())}\n"
                    f"Found examples: {sorted(y_raw.astype(str).unique().tolist())[:10]}\n\n"
                    f"Upload test CSV where '{target_col}' matches the same labels as training."
                )
            else:
                y_up = y_mapped.astype(int)

        #Only compute metrics if y_up is successfully created
        if y_up is not None:
            y_pred_up = preds_num  # numeric predictions

            from sklearn.metrics import confusion_matrix, classification_report

            cm_up = confusion_matrix(y_up, y_pred_up, labels=labels)
            fig2 = plot_confusion_matrix(cm_up, class_names)
            st.pyplot(fig2)

            rep = classification_report(y_up, y_pred_up, zero_division=0)
            st.text("Classification Report (Uploaded Test):")
            st.code(rep)


st.info("Tip: If deployment fails, check requirements.txt and Python version compatibility.")
