import pandas as pd
from datasets import load_dataset

def load_moltbook() -> pd.DataFrame:
    print("\nLoading FULL Moltbook dataset...\n")

    comments = load_dataset(
        "SimulaMet/moltbook-observatory-archive",
        "comments",
        split="archive"
    ).to_pandas().rename(columns={"id": "comment_id"})

    agents = load_dataset(
        "SimulaMet/moltbook-observatory-archive",
        "agents",
        split="archive"
    ).to_pandas()[["id", "name"]].rename(columns={"id": "agent_id", "name": "agent_name_y"})

    df = comments.merge(agents, on="agent_id", how="left").dropna(subset=["agent_name_y"])

    print("Rows:", len(df))
    print("Unique agents:", df["agent_name_y"].nunique())

    return df