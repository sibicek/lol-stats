import pandas as pd

# Load data
df = pd.read_csv("data/matches.csv")

# Basic overview
print("=== Dataset size ===")
print(df.shape)

print("\n=== First 5 rows ===")
print(df.head())

print("\n=== Column types ===")
print(df.dtypes)

# Missing values only concern rank columns (solo_tier, flex_tier) - unranked players
print("\n=== Missing values ===")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\n=== Basic statistics ===")
relevant_cols = ["duration", "kills", "deaths", "assists", "kda_ratio", 
                 "gold_per_min", "vision_score", "team_baronKills", 
                 "team_dragonKills", "team_towerKills"]
print(df[relevant_cols].describe())

print("\n=== Unique positions ===")
print(df["position"].value_counts())

print("\n=== Unique ranks ===")
# nan = unranked players - replaced with "UNRANKED" in preprocessing
print(df["solo_tier"].value_counts(dropna=False))

print("\n=== Win distribution ===")
print(df["win"].value_counts())
print(df["win"].value_counts(normalize=True))