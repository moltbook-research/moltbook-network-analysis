from datasets import load_dataset
import pandas as pd


def load_moltbook(sample_size=None):

    print("Loading Moltbook dataset...")

    # Load comments
    comments = load_dataset(
        "SimulaMet/moltbook-observatory-archive",
        "comments",
        split="archive"
    ).to_pandas()

    # Rename comment id immediately
    comments = comments.rename(columns={"id": "comment_id"})

    # Load agents
    agents = load_dataset(
        "SimulaMet/moltbook-observatory-archive",
        "agents",
        split="archive"
    ).to_pandas()

    # Keep only needed columns and rename cleanly
    agents = agents[["id", "name"]]
    agents = agents.rename(columns={
        "id": "agent_id_clean",
        "name": "agent_name"
    })

    # Merge on agent_id
    df = comments.merge(
        agents,
        left_on="agent_id",
        right_on="agent_id_clean",
        how="left"
    )

    print("Loaded rows:", len(df))
    print("COLUMNS AFTER MERGE:")
    print(df.columns)

    return df
