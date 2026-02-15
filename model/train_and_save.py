import os
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


ARTIFACT_DIR = os.path.join("model", "artifacts")


def to_dense_if_sparse(x):
    return x.toarray() if hasattr(x, "toarray") else x


def infer_task_type(y: pd.Series) -> str:
    return "binary" if y.nunique() == 2 else "multiclass"


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop"
    )


def get_models(task_type: str, random_state: int = 42):
    models = {}

    models["Logistic Regression"] = LogisticRegression(
        max_iter=2000,
        solver="liblinear"
    )

    models["Decision Tree"] = DecisionTreeClassifier(random_state=random_state)

    models["kNN"] = KNeighborsClassifier(n_neighbors=7)

    models["Naive Bayes"] = GaussianNB()

    models["Random Forest (Ensemble)"] = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1
    )

    if HAS_XGB:
        if task_type == "binary":
            models["XGBoost (Ensemble)"] = XGBClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=random_state,
                eval_metric="logloss",
                n_jobs=-1
            )
        else:
            models["XGBoost (Ensemble)"] = XGBClassifier(
                n_estimators=600,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=random_state,
                objective="multi:softprob",
                eval_metric="mlogloss",
                n_jobs=-1
            )
    else:
        print("WARNING: xgboost not installed. XGBoost model will be skipped.")

    return models


def compute_auc(task_type: str, y_true, y_proba, labels):
    if task_type == "binary":
        return roc_auc_score(y_true, y_proba)
    # multiclass: use OVR macro
    return roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=labels)


def evaluate(task_type: str, y_true, y_pred, y_proba, labels):
    avg = "binary" if task_type == "binary" else "macro"

    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": float(compute_auc(task_type, y_true, y_proba, labels)),
        "Precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "ConfusionMatrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "ClassificationReport": classification_report(y_true, y_pred, zero_division=0)
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    df = pd.read_csv(args.data)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found. Columns: {list(df.columns)}")

    # Drop fully empty columns if any
    df = df.dropna(axis=1, how="all")

    y_raw = df[args.target]
    X = df.drop(columns=[args.target])

    # For WDBC: diagnosis is 'M'/'B' -> convert to ints consistently
    if not np.issubdtype(y_raw.dtype, np.number):
        y_codes, uniques = pd.factorize(y_raw.astype(str))
        y = pd.Series(y_codes).astype(int)
        label_map = {int(i): str(lbl) for i, lbl in enumerate(uniques)}
    else:
        y = y_raw.astype(int)
        uniq = sorted(y.unique().tolist())
        label_map = {int(v): str(v) for v in uniq}

    task_type = infer_task_type(y)
    labels = sorted(y.unique().tolist())

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y if len(labels) > 1 else None
    )

    preprocessor = build_preprocessor(X_train)
    models = get_models(task_type, random_state=args.random_state)

    all_metrics = {}
    saved_models = []

    for name, clf in models.items():
        if name == "Naive Bayes":
            to_dense = FunctionTransformer(to_dense_if_sparse)
            pipe = Pipeline(steps=[
                ("prep", preprocessor),
                ("to_dense", to_dense),
                ("clf", clf)
            ])
        else:
            pipe = Pipeline(steps=[
                ("prep", preprocessor),
                ("clf", clf)
            ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        # Probabilities for AUC
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            proba = pipe.predict_proba(X_test)
            if task_type == "binary":
                y_proba = proba[:, 1]
            else:
                y_proba = proba
        else:
            # Fallback (rare)
            if hasattr(pipe.named_steps["clf"], "decision_function"):
                scores = pipe.decision_function(X_test)
                if task_type == "binary":
                    y_proba = 1 / (1 + np.exp(-scores))
                else:
                    exps = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                    y_proba = exps / np.sum(exps, axis=1, keepdims=True)
            else:
                y_proba = np.zeros(len(y_test)) if task_type == "binary" else np.zeros((len(y_test), len(labels)))

        metrics = evaluate(task_type, y_test, y_pred, y_proba, labels)
        all_metrics[name] = metrics

        safe_name = (
            name.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("/", "_")
        )
        model_path = os.path.join(ARTIFACT_DIR, f"{safe_name}.joblib")
        dump(pipe, model_path)

        saved_models.append({"name": name, "file": f"model/artifacts/{safe_name}.joblib"})
        print(f"Saved: {name} -> {model_path}")

    # Create summary table
    summary_rows = []
    for model_name, m in all_metrics.items():
        summary_rows.append({
            "ML Model Name": model_name,
            "Accuracy": round(m["Accuracy"], 4),
            "AUC": round(m["AUC"], 4),
            "Precision": round(m["Precision"], 4),
            "Recall": round(m["Recall"], 4),
            "F1": round(m["F1"], 4),
            "MCC": round(m["MCC"], 4),
        })

    metrics_summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": args.data,
        "target_column": args.target,
        "task_type": task_type,
        "labels": labels,
        "label_map": label_map,
        "models": saved_models,
        "table": summary_rows,
        "details": all_metrics
    }

    with open(os.path.join(ARTIFACT_DIR, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    with open(os.path.join(ARTIFACT_DIR, "label_info.json"), "w", encoding="utf-8") as f:
        json.dump({"task_type": task_type, "labels": labels, "label_map": label_map}, f, indent=2)

    print(f"Metrics saved to: {os.path.join(ARTIFACT_DIR, 'metrics_summary.json')}")


if __name__ == "__main__":
    main()
