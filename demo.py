#from TripleExtractor import TripleExtractor
from triple_extractor import TripleExtractor
from load_data import load_data
import numpy as np
from os import path
from tqdm import tqdm
import logging, sys

if __name__ == "__main__":
    logging.disable(sys.maxsize)
    te = TripleExtractor()
    if not path.exists("corenlp"):
        te.install()
    RELATION_MATRIX = te.import_FB15k_relations()

    train = load_data()
    train = train.iloc[:3]
    
    extracted_triples = []
    for i in tqdm(range(train.shape[0])):
        tmp_triples = []
        #try:
        te.get_doc(train.iloc[i].text)
        te.getValidTriples()
        te.set_experimental_relationship(RELATION_MATRIX)
        te.set_only_FB15K_valid_triples()
        te.set_prefered_ner()
        for triple in te.triples:
            tmp_triples.append([triple.subject, triple.relation, triple.object])
        if(tmp_triples != []):
            train.loc[i, "triple"] = [tmp_triples]

        # save each step
        train.to_csv("data_test.csv")
        #except:
        #    print(f"Exception at index {i}.")
