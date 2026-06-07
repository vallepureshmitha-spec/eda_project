import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# =====================================================================
# 1. DATA LOADING & PROFILE INSPECTION
# =====================================================================
file_path = "eda_raw_dataset.csv"

# Automatically generating a baseline dataset if the file is absent
if not os.path.exists(file_path):
    print(f"'{file_path}' not found. Generating a standard asset dataset for analysis...")
    np.random.seed(10)
    sample_data = {
        "Transaction_Amount": np.random.normal(loc=500, scale=150, size=100).tolist() + [5000, 5500], # Includes skews
        "Customer_Age": np.random.randint(18, 70, size=102).tolist(),
        "Account_Balance": np.random.uniform(1000, 50000, size=102).tolist(),
        "Region": np.random.choice(["North", "East", "South", "West"], size=102).tolist(),
        "Device_Type": np.random.choice(["Mobile", "Desktop", "Tablet"], size=102).tolist()
    }
    # Introduce some artificial missing entries for EDA discovery
    df_generation = pd.DataFrame(sample_data)
    df_generation.loc[df_generation.sample(frac=0.05).index, "Customer_Age"] = np.nan
    df_generation.to_csv(file_path, index=False)

# Load dataset
df = pd.read_csv(file_path)

print("=== 1. DATASET COMPOSITION ===")
print(f"Total Observations (Rows): {df.shape[0]}")
print(f"Total Variables (Columns): {df.shape[1]}")
print("\nVariable Configurations:")
print(df.dtypes)

# =====================================================================
# 2. STATISTICAL SUMMARIES
# =====================================================================
print("\n=== 2. MATHEMATICAL & STATISTICAL SUMMARIES ===")
print("\nQuantitative Variable Metrics:")
print(df.describe())

print("\nQualitative Variable Cardinality:")
print(df.describe(include=["object"]))

print("\nMissing Data Matrix:")
missing_counts = df.isnull().sum()
print(missing_counts[missing_counts > 0] if missing_counts.sum() > 0 else "No missing values detected.")

# =====================================================================
# 3. DISTRIBUTION & TREND VISUALIZATION
# =====================================================================
print("\n=== 3. GENERATING EXPLORATORY GRAPHICAL REPORTS ===")
sns.set_theme(style="white")
fig = plt.figure(figsize=(16, 12))

# Subplot 1: Quantitative Density Distribution
plt.subplot(2, 2, 1)
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
primary_numeric = numeric_columns[0]
sns.histplot(df[primary_numeric], kde=True, color="darkslateblue", bins=20)
plt.title(f"Density Distribution Analysis: {primary_numeric}", fontsize=12, fontweight="bold")
plt.xlabel(primary_numeric)
plt.ylabel("Frequency Density")

# Subplot 2: Categorical Stratification (Boxplot)
plt.subplot(2, 2, 2)
categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
if len(categorical_columns) > 0 and len(numeric_columns) > 0:
    primary_category = categorical_columns[0]
    sns.boxplot(x=df[primary_category], y=df[primary_numeric], palette="Set2", hue=df[primary_category], legend=False)
    plt.title(f"{primary_numeric} Variance Across {primary_category}", fontsize=12, fontweight="bold")
    plt.xlabel(primary_category)
    plt.ylabel(primary_numeric)

# Subplot 3: Multi-Variable Interaction (Scatter Plot)
plt.subplot(2, 2, 3)
if len(numeric_columns) >= 2:
    secondary_numeric = numeric_columns[1]
    sns.scatterplot(x=df[primary_numeric], y=df[secondary_numeric], alpha=0.7, color="crimson")
    plt.title(f"Bivariate Interaction: {primary_numeric} vs {secondary_numeric}", fontsize=12, fontweight="bold")
    plt.xlabel(primary_numeric)
    plt.ylabel(secondary_numeric)

# Subplot 4: Linear Correlation Heatmap
plt.subplot(2, 2, 4)
if len(numeric_columns) >= 2:
    correlation_matrix = df[numeric_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="mako", fmt=".2f", vmin=-1, vmax=1, cbar=True)
    plt.title("Linear Correlation Coefficient Matrix", fontsize=12, fontweight="bold")

plt.tight_layout()
print("Exhibiting analytical plots. Dismiss the interactive viewer to terminate the program.")
plt.show()
