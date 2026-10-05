# ==============================
# CUSTOMER CHURN ANALYSIS
# Major Project - 2
# ==============================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("customer_churn.csv")

# ==============================
# DATA LOADING
# ==============================

print("\nFIRST 5 RECORDS")
print(df.head())

print("\nLAST 5 RECORDS")
print(df.tail())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns)

# ==============================
# DATA EXPLORATION & CLEANING
# ==============================

print("\nDATA TYPES")
print(df.dtypes)

print("\nSTATISTICAL INFORMATION")
print(df.describe())

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE RECORDS")
print(df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

# TotalCharges cleanup
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"].fillna(
    df["TotalCharges"].median(),
    inplace=True
)

# ==============================
# TENURE GROUPS
# ==============================

def tenure_group(x):
    if x <= 12:
        return "0-12 Months"
    elif x <= 24:
        return "13-24 Months"
    elif x <= 48:
        return "25-48 Months"
    elif x <= 60:
        return "49-60 Months"
    else:
        return "60+ Months"

df["Tenure_Group"] = df["tenure"].apply(tenure_group)

print("\nUNIQUE VALUES")
for col in df.columns:
    print(f"\n{col}")
    print(df[col].nunique())

print("\nCLEANED DATASET")
print(df.head())

# ==============================
# DATA ANALYSIS
# ==============================

total_customers = len(df)

churned_customers = len(
    df[df["Churn"] == "Yes"]
)

retained_customers = len(
    df[df["Churn"] == "No"]
)

churn_rate = (
    churned_customers /
    total_customers
) * 100

avg_monthly_charges = df["MonthlyCharges"].mean()

avg_total_charges = df["TotalCharges"].mean()

avg_tenure = df["tenure"].mean()

print("\nTOTAL CUSTOMERS:", total_customers)
print("CHURNED CUSTOMERS:", churned_customers)
print("RETAINED CUSTOMERS:", retained_customers)
print("CHURN RATE:", round(churn_rate,2),"%")

print("AVG MONTHLY CHARGES:",
      round(avg_monthly_charges,2))

print("AVG TOTAL CHARGES:",
      round(avg_total_charges,2))

print("AVG TENURE:",
      round(avg_tenure,2))

# ==============================
# GROUP ANALYSIS
# ==============================

print("\nCUSTOMERS BY GENDER")
print(df["gender"].value_counts())

print("\nCUSTOMERS BY CONTRACT")
print(df["Contract"].value_counts())

print("\nCUSTOMERS BY INTERNET SERVICE")
print(df["InternetService"].value_counts())

print("\nCUSTOMERS BY PAYMENT METHOD")
print(df["PaymentMethod"].value_counts())

# ==============================
# CHURN RATE ANALYSIS
# ==============================

print("\nCHURN RATE BY GENDER")
print(
    pd.crosstab(
        df["gender"],
        df["Churn"],
        normalize="index"
    ) * 100
)

print("\nCHURN RATE BY CONTRACT")
print(
    pd.crosstab(
        df["Contract"],
        df["Churn"],
        normalize="index"
    ) * 100
)

print("\nCHURN RATE BY INTERNET SERVICE")
print(
    pd.crosstab(
        df["InternetService"],
        df["Churn"],
        normalize="index"
    ) * 100
)

print("\nCHURN RATE BY PAYMENT METHOD")
print(
    pd.crosstab(
        df["PaymentMethod"],
        df["Churn"],
        normalize="index"
    ) * 100
)

# ==============================
# TOP 10 CUSTOMERS
# ==============================

print("\nTOP 10 CUSTOMERS BY TOTAL CHARGES")

print(
    df.sort_values(
        by="TotalCharges",
        ascending=False
    )
    [["customerID",
      "TotalCharges"]]
    .head(10)
)

# ==============================
# VISUALIZATIONS
# ==============================

sns.set_style("whitegrid")

# 1 Pie Chart
plt.figure(figsize=(6,6))
df["Churn"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)
plt.title("Churn Distribution")
plt.ylabel("")
plt.show()

# 2 Contract Distribution
plt.figure(figsize=(8,5))
sns.countplot(
    data=df,
    x="Contract"
)
plt.title("Customer Distribution by Contract")
plt.show()

# 3 Churn by Contract
plt.figure(figsize=(8,5))
sns.countplot(
    data=df,
    x="Contract",
    hue="Churn"
)
plt.title("Churn by Contract")
plt.show()

# 4 Churn by Gender
plt.figure(figsize=(6,5))
sns.countplot(
    data=df,
    x="gender",
    hue="Churn"
)
plt.title("Churn by Gender")
plt.show()

# 5 Churn by Internet Service
plt.figure(figsize=(8,5))
sns.countplot(
    data=df,
    x="InternetService",
    hue="Churn"
)
plt.title("Churn by Internet Service")
plt.show()

# 6 Churn by Payment Method
plt.figure(figsize=(12,5))
sns.countplot(
    data=df,
    x="PaymentMethod",
    hue="Churn"
)
plt.xticks(rotation=20)
plt.title("Churn by Payment Method")
plt.show()

# 7 Tenure Distribution
plt.figure(figsize=(8,5))
sns.histplot(
    df["tenure"],
    bins=30
)
plt.title("Tenure Distribution")
plt.show()

# 8 Monthly Charges Distribution
plt.figure(figsize=(8,5))
sns.histplot(
    df["MonthlyCharges"],
    bins=30
)
plt.title("Monthly Charges Distribution")
plt.show()

# 9 Scatter Plot
plt.figure(figsize=(8,5))
sns.scatterplot(
    data=df,
    x="tenure",
    y="MonthlyCharges",
    hue="Churn"
)
plt.title("Tenure vs Monthly Charges")
plt.show()

# 10 Box Plot
plt.figure(figsize=(8,5))
sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)
plt.title("Monthly Charges by Churn")
plt.show()

# 11 Total Charges by Churn
plt.figure(figsize=(8,5))
sns.boxplot(
    data=df,
    x="Churn",
    y="TotalCharges"
)
plt.title("Total Charges by Churn")
plt.show()

# 12 Churn Rate by Tenure Group
plt.figure(figsize=(8,5))
sns.countplot(
    data=df,
    x="Tenure_Group",
    hue="Churn"
)
plt.title("Churn by Tenure Group")
plt.show()

# ==============================
# CORRELATION HEATMAP
# ==============================

numeric_cols = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

plt.figure(figsize=(8,6))

sns.heatmap(
    df[numeric_cols].corr(),
    annot=True
)

plt.title("Correlation Heatmap")
plt.show()

# ==============================
# EXPORT FOR POWER BI
# ==============================

df.to_csv(
    "customer_churn_cleaned.csv",
    index=False
)

print("\nCLEANED DATASET EXPORTED")