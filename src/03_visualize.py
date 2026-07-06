import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/teams.csv")

# --- Correlation matrix ---

# list of variables to analyze
features = ["avg_kda", "avg_gold_per_min", "avg_damage_champ_per_min", 
            "avg_vision_score", "baron_kills", "dragon_kills", "tower_kills", "win"]

corr = df[features].corr()  # correlation between all pairs of variables

# Plot correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation matrix")
plt.tight_layout()
plt.savefig("report/correlation_matrix.png")
plt.show()

# --- Boxplots: distribution of each variable by win/loss ---

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

# same variables as above, excluding "win" (used as x axis)
features_only = ["avg_kda", "avg_gold_per_min", "avg_damage_champ_per_min", 
                 "avg_vision_score", "baron_kills", "dragon_kills", "tower_kills"]

for i, feature in enumerate(features_only):
    sns.boxplot(data=df, x="win", y=feature, ax=axes[i])
    axes[i].set_title(feature)
    axes[i].set_xlabel("Win")

axes[-1].set_visible(False)

plt.tight_layout()
plt.savefig("report/boxplots.png")
plt.show()