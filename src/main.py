"""
data_prep.py
Loads the credit card transaction dataset and performs initial
exploratory data analysis (EDA) to understand the class imbalance
and feature distributions before modeling.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
# 3. Class distribution (the key challenge: imbalance)
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
# Run everything
# ---------------------------------------------------------
if __name__ == "__main__":
    df = load_data()
    basic_info(df)
    class_distribution(df)
    plot_class_distribution(df)
    plot_amount_comparison(df)
    X_train, X_test, y_train, y_test = preprocess_data(df)