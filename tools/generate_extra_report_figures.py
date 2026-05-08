import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/bc_prepared.csv"
FIG_DIR = "reports/figures"

os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Create Status column for readable plots
df["Status"] = df["event"].map({0: "Alive", 1: "Dead"})


# ---------------- Figure 4: Outcome and Survival Months ---------------- #

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.countplot(data=df, x="Status")
plt.title("Outcome Distribution (Alive vs Dead)")
plt.xlabel("Status")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
sns.histplot(df["time"], bins=30, kde=True)
plt.title("Survival Months Distribution")
plt.xlabel("Survival Months")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "Figure4_outcome_survival_distribution.png"), dpi=300)
plt.close()


# ---------------- Figure 5: Age and Tumor Size ---------------- #

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df["Age"], bins=30, kde=True)
plt.title("Age Distribution of Patients")
plt.xlabel("Age")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
sns.boxplot(data=df, x="Status", y="Tumor Size")
plt.title("Tumor Size by Outcome")
plt.xlabel("Status")
plt.ylabel("Tumor Size")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "Figure5_age_tumor_by_outcome.png"), dpi=300)
plt.close()

print("Saved extra report figures to reports/figures")