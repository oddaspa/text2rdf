from TripleExtractor import TripleExtractor
import re
import pandas as pd
import numpy as np
from os import path
from tqdm import tqdm
import logging, sys

def clean_text(text):
    return re.sub("[^A-Za-z0-9 .!?,]", "", text)

def get_spo(triple):
    return triple.subject + ", " + triple.relation + ", " + triple.object

if __name__ == "__main__":
    logging.disable(sys.maxsize)
    te = TripleExtractor()

    if not path.exists("corenlp"):
        te.install()
    te.import_FB15k_relations()

    train = pd.read_csv("fake-news/train.csv")
    train.drop(train[train['text'].isna()].index, inplace=True)
    train['text'] = train['text'].apply(lambda text: clean_text(text))

    # initialize empty
    train['triple'] = ["NaN"] * train.shape[0]
    
    extracted_triples = []
    for i in tqdm(range(train.shape[0])):
        tmp_triples = []
        try:
            te.get_doc(train.iloc[i].text)
            te.getValidTriples()
            te.set_experimental_relationship()
            te.set_only_FB15K_valid_triples()
            te.set_prefered_ner()
            for triple in te.triples:
                tmp_triples.append([triple.subject, triple.relation, triple.object])
            if(tmp_triples != []):
                train.loc[i, "triple"] = [tmp_triples]
    
            # save each step
            train.to_csv("data_test.csv")
        except:
            print(f"Exception at index {i}.")
