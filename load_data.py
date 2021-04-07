import pandas as pd
import re

def clean_text(text):
    return re.sub("[^A-Za-z0-9 .!?,]", "", text)


def load_data(path, START_SLICE, END_SLICE):
    df = pd.read_csv(path)
    df.drop(df[df["text"].isna()].index, inplace=True)
    df["text"] = df["text"].apply(lambda text: clean_text(text))

    # initialize empty
    df["triple"] = ["NaN"] * df.shape[0]
    if START_SLICE == -1:
        return df.iloc[0:2]
    return df.iloc[START_SLICE:END_SLICE]
