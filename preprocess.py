import pandas as pd

def load_data():
    fake = pd.read_csv("data/Fake.csv")
    real = pd.read_csv("data/True.csv")

    fake["label"] = 0
    real["label"] = 1

    df = pd.concat([fake, real])
    df = df.sample(frac=1, random_state=42)

    X = df["text"]
    y = df["label"]

    return X, y