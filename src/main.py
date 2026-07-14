"""
main.py
Loads the credit card transaction dataset, explores it, preprocesses it,
and trains/evaluates three models (Logistic Regression, Random Forest,
and XGBoost) to detect fraud.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------
DATA_PATH = "data/creditcard.csv"

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print(f"Loaded dataset with shape: {df.shape}")
    return df


# ---------------------------------------------------------
# 2. Basic info & missing values
# ---------------------------------------------------------
def basic_info(df):
    print("\n--- Data types & missing values ---")
    print(df.info())
    print("\nMissing values per column:")
    print(df.isnull().sum().sum(), "total missing values")


# ---------------------------------------------------------
# 3. Class distribution
# ---------------------------------------------------------
def class_distribution(df):
    counts = df["Class"].value_counts()
    percentages = df["Class"].value_counts(normalize=True) * 100
    print("\n--- Class distribution ---")
    print(f"Legit (0): {counts[0]}  ({percentages[0]:.3f}%)")
    print(f"Fraud (1): {counts[1]}  ({percentages[1]:.3f}%)")


# ---------------------------------------------------------
# 4. Visualize the imbalance
# ---------------------------------------------------------
def plot_class_distribution(df):
    plt.figure(figsize=(5, 4))
    sns.countplot(x="Class", data=df)
    plt.title("Class Distribution (0 = Legit, 1 = Fraud)")
    plt.yscale("log")
    plt.xlabel("Class")
    plt.ylabel("Count (log scale)")
    plt.savefig("class_distribution.png")
    plt.close()
    print("Saved class_distribution.png")


# ---------------------------------------------------------
# 5. Compare Amount: fraud vs legit
# ---------------------------------------------------------
def plot_amount_comparison(df):
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Class", y="Amount", data=df)
    plt.title("Transaction Amount by Class")
    plt.yscale("log")
    plt.xlabel("Class (0 = Legit, 1 = Fraud)")
    plt.ylabel("Amount (log scale)")
    plt.savefig("amount_comparison.png")
    plt.close()
    print("Saved amount_comparison.png")


# ---------------------------------------------------------
# 6. Preprocess: scale + split
# ---------------------------------------------------------
def preprocess_data(df):
    df = df.copy()

    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])
    df["Time"] = scaler.fit_transform(df[["Time"]])

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("\n--- Preprocessing complete ---")
    print(f"Training set: {X_train.shape[0]} rows")
    print(f"Test set: {X_test.shape[0]} rows")
    print(f"Fraud % in train: {y_train.mean()*100:.3f}%")
    print(f"Fraud % in test: {y_test.mean()*100:.3f}%")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------
# 7a. Train a baseline Logistic Regression model
# ---------------------------------------------------------
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("\n--- Logistic Regression trained ---")
    return model


# ---------------------------------------------------------
# 7b. Train a Random Forest model
# ---------------------------------------------------------
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("\n--- Random Forest trained ---")
    return model


# ---------------------------------------------------------
# 7c. Train an XGBoost model
# ---------------------------------------------------------
def train_xgboost(X_train, y_train):
    # scale_pos_weight tells XGBoost how imbalanced the classes are
    # (roughly: count of legit / count of fraud)
    fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=100,
        scale_pos_weight=fraud_ratio,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("\n--- XGBoost trained ---")
    return model


# ---------------------------------------------------------
# 8. Evaluate a model
# ---------------------------------------------------------
def evaluate_model(model, X_test, y_test, model_name="model"):
    y_pred = model.predict(X_test)

    print(f"\n--- Classification Report ({model_name}) ---")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    print(f"--- Confusion Matrix ({model_name}) ---")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")


# ---------------------------------------------------------
# Run everything
# ---------------------------------------------------------
if __name__ == "__main__":
    df = load_data()
    basic_info(df)
    class_distribution(df)
    plot_class_distribution(df)
    plot_amount_comparison(df)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Logistic Regression
    logreg_model = train_logistic_regression(X_train, y_train)
    evaluate_model(logreg_model, X_test, y_test, model_name="Logistic Regression")

    # Random Forest
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test, model_name="Random Forest")

    # XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    evaluate_model(xgb_model, X_test, y_test, model_name="XGBoost")