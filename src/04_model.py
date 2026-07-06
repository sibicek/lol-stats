import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

df = pd.read_csv("data/teams.csv")
y = df["win"]

# Model 1 - individual player statistics
features_individual = ["avg_kda", "avg_gold_per_min", "avg_vision_score"]
X1 = sm.add_constant(df[features_individual])
model1 = sm.Logit(y, X1).fit()
print("\n=== Model 1: Individual statistics ===")
print(model1.summary())

# Model 2 - team objectives
features_objectives = ["baron_kills", "dragon_kills", "tower_kills"]
X2 = sm.add_constant(df[features_objectives])
model2 = sm.Logit(y, X2).fit()
print("\n=== Model 2: Team objectives ===")
print(model2.summary())

# AUC comparison
auc1 = roc_auc_score(y, model1.predict(X1))
auc2 = roc_auc_score(y, model2.predict(X2))
print(f"\nAUC Model 1 (individual): {auc1:.3f}")
print(f"AUC Model 2 (objectives): {auc2:.3f}")

# ROC curves
fpr1, tpr1, _ = roc_curve(y, model1.predict(X1))
fpr2, tpr2, _ = roc_curve(y, model2.predict(X2))

plt.figure(figsize=(8, 6))
plt.plot(fpr1, tpr1, label=f"Individual (AUC = {auc1:.3f})")
plt.plot(fpr2, tpr2, label=f"Objectives (AUC = {auc2:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve - model comparison")
plt.legend()
plt.tight_layout()
plt.savefig("report/roc_curve.png")
plt.show()