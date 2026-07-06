import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Load raw per-player data (not aggregated)
df = pd.read_csv("data/matches.csv")
df = df[df["position"] != "Invalid"]
df["win"] = df["win"].map({True: 1, False: 0, "TRUE": 1, "FALSE": 0})

positions = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "SUPPORT"]
features = ["kda_ratio", "gold_per_min", "vision_score"]

results = []

for pos in positions:
    subset = df[df["position"] == pos]
    X = sm.add_constant(subset[features])
    y = subset["win"]
    
    model = sm.Logit(y, X).fit(disp=0)
    auc = roc_auc_score(y, model.predict(X))
    
    results.append({
        "position": pos,
        "auc": auc,
        "kda_coef": model.params["kda_ratio"],
        "gold_coef": model.params["gold_per_min"],
        "vision_coef": model.params["vision_score"],
    })
    
    print(f"\n=== {pos} ===")
    print(model.summary())

# Summary table
results_df = pd.DataFrame(results)
print("\n=== Coefficient summary by position ===")
print(results_df)

# Coefficient plot by position
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (coef, title) in enumerate([
    ("kda_coef", "KDA ratio"),
    ("gold_coef", "Gold per min"),
    ("vision_coef", "Vision score")
]):
    axes[i].bar(results_df["position"], results_df[coef])
    axes[i].set_title(title)
    axes[i].set_xlabel("Position")
    axes[i].set_ylabel("Coefficient")
    axes[i].axhline(0, color="black", linewidth=0.8, linestyle="--")

plt.tight_layout()
plt.savefig("report/by_position.png")
plt.show()