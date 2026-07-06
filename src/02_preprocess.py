import pandas as pd

df = pd.read_csv("data/matches.csv")

# Remove invalid rows
df = df[df["position"] != "Invalid"]
# nan -> UNRANKED
df["solo_tier"] = df["solo_tier"].fillna("UNRANKED")

# Convert win to 0/1
df["win"] = df["win"].map({True: 1, False: 0, "TRUE": 1, "FALSE": 0})

# Aggregate to team level (game_id + win as key)
team_df = df.groupby(["game_id", "win"]).agg(
    duration=("duration", "first"),
    avg_kda=("kda_ratio", "mean"),
    avg_gold_per_min=("gold_per_min", "mean"),
    avg_damage_champ_per_min=("damage_champ_per_min", "mean"),
    avg_vision_score=("vision_score", "mean"),
    baron_kills=("team_baronKills", "first"),
    dragon_kills=("team_dragonKills", "first"),
    tower_kills=("team_towerKills", "first"),
).reset_index()

print(team_df.shape)
print(team_df.head())
print(team_df["win"].value_counts())

team_df.to_csv("data/teams.csv", index=False)
print("Saved to data/teams.csv")