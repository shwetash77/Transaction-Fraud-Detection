"""
data_prep.py
Loads the credit card transaction dataset and performs initial
exploratory data analysis (EDA) to understand the class imbalance
and feature distributions before modeling.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
# Run everything
# ---------------------------------------------------------
if __name__ == "__main__":
    df = load_data()
    basic_info(df)
    class_distribution(df)