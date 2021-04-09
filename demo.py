#from TripleExtractor import TripleExtractor
from triple_extractor import TripleExtractor
from load_data import load_data
import numpy as np
from tqdm import tqdm
import os, logging, sys
from uuid import uuid4
if __name__ == "__main__":
    file_name = sys.argv[0]
    try:
        NAME = str(sys.argv[1])
        START_SLICE = int(sys.argv[2])
        END_SLICE = int(sys.argv[-1])
        print(f"slice from {START_SLICE} to {END_SLICE}")
    except:
        print(f"WRONG FORMAT OF SLICES. SHOULD BE INTEGERS BUT GOT: {sys.argv[2]} and {sys.argv[-1]}")
    logging.disable(sys.maxsize)

    cpu_count = os.cpu_count()

    te = TripleExtractor(threads=cpu_count)
    if not os.path.exists("corenlp"):
        te.install()
    RELATION_MATRIX = te.import_FB15k_relations()

    path_dataset = "./fakenewsnet/"
    path_triple_dataset = "./fakenewsnet_triples"
    tag = ".csv"
    data = load_data(path_dataset + NAME + tag, START_SLICE, END_SLICE)

    UNIQUE_ID = str(uuid4())
    if not os.path.exists(path_triple_dataset):
        os.mkdir(path_triple_dataset)
    
    dataset_name = os.path.join(path_triple_dataset, NAME + "_triple_" + UNIQUE_ID + tag)

    for i in tqdm(range(data.shape[0])):
        tmp_triples = []
        try:
            te.get_doc(data.iloc[i].text)
            te.getValidTriples()
            te.set_experimental_relationship(RELATION_MATRIX)
            te.set_only_FB15K_valid_triples()
            te.set_prefered_ner()
            for triple in te.triples:
                tmp_triples.append([triple.subject, triple.relation, triple.object])
            if(tmp_triples != []):
                data.loc[i, "triple"] = [tmp_triples]

            # save each step
            data.to_csv(dataset_name)
        except:
            print(f"Exception at index {i}.")
