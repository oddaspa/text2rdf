from TripleExtractor import TripleExtractor
import re
import pandas as pd
import numpy as np


def clean_text(text):
    return re.sub("[^A-Za-z0-9 .!?,]", "", text)

def get_spo(triple):
    return triple.subject + ", " + triple.relation + ", " + triple.object

if __name__ == "__main__":
    te = TripleExtractor()
    te.install()
    te.import_FB15k_relations()

    train = pd.read_csv("fake-news/train.csv")
    test_pd = train.iloc[0:3]
    clean_test = test_pd

    clean_test['text'] = clean_test['text'].apply(lambda text: clean_text(text))
    
    extracted_triples = []
    entities = []
    for i in range(clean_test.shape[0]):
        tmp_triples = []
        te.get_doc(clean_test.iloc[i].text)
        te.getValidTriples()
        te.set_experimental_relationship()
        te.set_only_FB15K_valid_triples()
        te.set_prefered_ner()
        for triple in te.triples:
            tmp_triples.append(", ".join([triple.subject, triple.relation, triple.object]))
            entities.append(triple.subject)
            entities.append(triple.object)
        extracted_triples.append(tmp_triples)
    
    clean_test['triple'] = extracted_triples

    clean_test.to_csv("clean_test.csv")
    entities = np.unique(entities)
    entities = pd.Series(entities)
    entities.to_csv("entities.csv")
