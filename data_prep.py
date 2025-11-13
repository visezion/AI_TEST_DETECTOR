from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split


def _split_to_frame(split):
    """Flatten QA records into (text, label) rows for human vs. AI."""
    rows = []
    for sample in split:
        ai_text = sample.get("text")
        human_text = sample.get("human_response")
        if isinstance(ai_text, str) and ai_text.strip():
            rows.append({"text": ai_text.strip(), "label": 1})
        if isinstance(human_text, str) and human_text.strip():
            rows.append({"text": human_text.strip(), "label": 0})
    return pd.DataFrame(rows)


def load_ai_human_dataset():
    dataset = load_dataset("pszemraj/HC3-textgen-qa")
    df = pd.concat(
        [_split_to_frame(dataset["train"]), _split_to_frame(dataset["test"])],
        ignore_index=True,
    )

    df = df.dropna(subset=["text", "label"])

    train_df, val_df = train_test_split(
        df, test_size=0.1, random_state=42, stratify=df["label"]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
