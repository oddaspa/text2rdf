import pandas as pd
import re

def clean_text(text):
    return re.sub("[^A-Za-z0-9 .!?,]", "", text)


def load_data():
    train = pd.read_csv("fake-news/train.csv")
    train.drop(train[train["text"].isna()].index, inplace=True)
    train["text"] = train["text"].apply(lambda text: clean_text(text))
    train["text"] = train["text"].apply(lambda text: text[:1000])
    # initialize empty
    train["triple"] = ["NaN"] * train.shape[0]
    return train
