from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)

# 1. Set up project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "raw" / "wdbc.data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 2. Define the dataset columns
features = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

columns = ["id", "diagnosis"] + features


def main():
    # 3. Load the actual dataset
    if not DATA_FILE.exists():
        print("ERROR: Dataset not found!")
        print("Expected location:", DATA_FILE)
        print("Put wdbc.data inside data/raw/")
        return

    df = pd.read_csv(DATA_FILE, header=None, names=columns)

    if df.shape[1] != 32:
        print("ERROR: The dataset should contain 32 columns.")
        print("Check that you downloaded wdbc.data, not wdbc.names.")
        return

    # 4. Prepare and clean the data
    df["diagnosis"] = (
        df["diagnosis"].astype(str).str.strip().str.upper()
    )

    df = df[df["diagnosis"].isin(["B", "M"])].copy()

    for column in features:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=features + ["diagnosis"])

    df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})

    print("\nDISEASE PREDICTION PROJECT")
    print("=" * 45)
    print("Dataset loaded successfully!")
    print("Number of records:", len(df))
    print("Number of features:", len(features))
    print("\nDiagnosis counts:")
    print(df["diagnosis"].map({
        0: "Benign",
        1: "Malignant"
    }).value_counts())

    # 5. Visualize the diagnosis distribution
    sns.set_theme(style="whitegrid")

    diagnosis_names = df["diagnosis"].map({
        0: "Benign",
        1: "Malignant"
    })

    plt.figure(figsize=(7, 5))
    sns.countplot(x=diagnosis_names)
    plt.title("Distribution of Diagnosis Labels")
    plt.xlabel("Diagnosis")
    plt.ylabel("Number of Records")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_diagnosis_distribution.png")
    plt.show()
    plt.close()

    # 6. Visualize selected feature distributions
    selected_features = [
        "radius_mean",
        "texture_mean",
        "area_mean",
        "smoothness_mean"
    ]

    for feature in selected_features:
        plt.figure(figsize=(7, 5))
        sns.histplot(
            data=df,
            x=feature,
            hue=diagnosis_names,
            kde=True
        )
        plt.title(f"Distribution of {feature}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{feature}_distribution.png")
        plt.show()
        plt.close()

    # 7. Create a correlation heatmap
    plt.figure(figsize=(12, 9))
    sns.heatmap(
        df[features[:10]].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )
    plt.title("Correlation Heatmap of Selected Features")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_correlation_heatmap.png")
    plt.show()
    plt.close()

    # 8. Separate features and target
    X = df[features]
    y = df["diagnosis"].astype(int)

    # 9. Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))

    # 10. Define the machine learning models
    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000))
        ]),

        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            ))
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=200,
                random_state=42
            ))
        ])
    }

    results = []
    trained_models = {}

    # 11. Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        scores = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1-score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "ROC-AUC": roc_auc_score(y_test, probabilities)
        }

        results.append(scores)
        trained_models[name] = model

        print("\nClassification report:")
        print(classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Benign", "Malignant"],
            zero_division=0
        ))

        # Save a confusion matrix for each model
        matrix = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        )

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"]
        )
        plt.title(f"{name} - Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.tight_layout()

        filename = name.lower().replace(" ", "_")
        plt.savefig(OUTPUT_DIR / f"{filename}_confusion_matrix.png")
        plt.show()
        plt.close()

    # 12. Compare model performance
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="ROC-AUC",
        ascending=False
    )

    results_df.to_csv(
        OUTPUT_DIR / "model_results.csv",
        index=False
    )

    print("\nMODEL PERFORMANCE COMPARISON")
    print("=" * 70)
    print(results_df.round(4).to_string(index=False))

    # 13. Visualize ROC curves
    plt.figure(figsize=(8, 6))

    for name, model in trained_models.items():
        probabilities = model.predict_proba(X_test)[:, 1]

        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_test, probabilities
        )

        auc_value = roc_auc_score(y_test, probabilities)

        plt.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{name} (AUC={auc_value:.3f})"
        )

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_roc_curve_comparison.png")
    plt.show()
    plt.close()

    # 14. Visualize model comparison
    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC"
    ]

    results_df.set_index("Model")[metrics].plot(
        kind="bar",
        figsize=(11, 6),
        ylim=(0, 1.05)
    )

    plt.title("Machine Learning Model Comparison")
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_model_comparison.png")
    plt.show()
    plt.close()

    # 15. Save the best model based on test ROC-AUC
    best_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_name]

    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")

    print("\nBest model by ROC-AUC:", best_name)
    print("Results saved in:", OUTPUT_DIR)
    print("Best model saved in:", MODEL_DIR)
    print("\nPROJECT COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()