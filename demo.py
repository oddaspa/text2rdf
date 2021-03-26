from TripleExtractor import TripleExtractor
import re
import pandas as pd
import numpy as np
import os
from os import path
from tqdm import tqdm
import logging, sys, threading
from multiprocessing import Pool


def clean_text(text):
    return re.sub("[^A-Za-z0-9 .!?,]", "", text)

def get_spo(triple):
    return triple.subject + ", " + triple.relation + ", " + triple.object

def extract_triple(text):
    extractor = TripleExtractor()
    extractor.import_FB15k_relations()
    extractor.get_doc(text)
    extractor.getValidTriples()
    extractor.set_experimental_relationship()
    extractor.set_only_FB15K_valid_triples()
    extractor.set_prefered_ner()
    tmp_triples = []

    for triple in extractor.triples:
        tmp_triples.append([triple.subject, triple.relation, triple.object])
    return tmp_triples


if __name__ == "__main__":
    # Multiprocessing
    pool = Pool(os.cpu_count())
    # remove INFO logging
    logging.disable(sys.maxsize)

    # initialize extractor
    te = TripleExtractor()
    if not path.exists("corenlp"):
        te.install()
    te.import_FB15k_relations()

    train = pd.read_csv("fake-news/train.csv")
    train = train.iloc[:10]
    train.drop(train[train['text'].isna()].index, inplace=True)
    train['text'] = train['text'].apply(lambda text: clean_text(text))

    # initialize empty
    train['triple'] = ["NaN"] * train.shape[0]
    inputs = train['text']
    
    extracted_triples = pool.map(extract_triple, inputs)
    train["triple"] = extract_triple
    
    train.to_csv("data_test.csv")
