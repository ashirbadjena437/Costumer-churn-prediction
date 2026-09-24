import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "Telco-Customer-Churn.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "churn_model.pkl"
)


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# 3. Data cleaning
# --------------------------------------------------

print("\nCleaning data...")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove rows with missing values
df.dropna(inplace=True)

# Remove customer ID
df.drop(
    "customerID",
    axis=1,
    inplace=True
)

print("Data cleaning completed.")
print("Cleaned dataset shape:", df.shape)


# --------------------------------------------------
# 4. Convert target variable
# --------------------------------------------------

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})


# --------------------------------------------------
# 5. Separate features and target
# --------------------------------------------------

X = df.drop(
    "Churn",
    axis=1
)

y = df["Churn"]


# --------------------------------------------------
# 6. Identify columns
# --------------------------------------------------

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns


# --------------------------------------------------
# 7. Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# --------------------------------------------------
# 8. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 9. Create Logistic Regression pipeline
# --------------------------------------------------

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


# --------------------------------------------------
# 10. Train model
# --------------------------------------------------

print("\nTraining Logistic Regression model...")

model.fit(
    X_train,
    y_train
)

print("Model training completed.")


# --------------------------------------------------
# 11. Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# --------------------------------------------------
# 12. Evaluation
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print(f"Accuracy :  {accuracy:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall   :  {recall:.4f}")
print(f"F1 Score :  {f1:.4f}")
print(f"ROC-AUC  :  {roc_auc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Churn",
            "Churn"
        ]
    )
)


# --------------------------------------------------
# 13. Save model
# --------------------------------------------------

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_PATH
)

print("\n" + "=" * 50)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 50)

print("Model location:")
print(MODEL_PATH)
