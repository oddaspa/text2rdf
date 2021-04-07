import pandas as pd
import re

def clean_text(text):
    return re.sub("[^A-Za-z0-9 .!?,]", "", text)


def load_data(path, START_SLICE, END_SLICE):
    train = pd.read_csv(path)
    train.drop(train[train["text"].isna()].index, inplace=True)
    train["text"] = train["text"].apply(lambda text: clean_text(text))
    #train["text"] = train["text"].apply(lambda text: text[:1000])
    # initialize empty
    train["triple"] = ["NaN"] * train.shape[0]
    return train.iloc[START_SLICE:END_SLICE]
