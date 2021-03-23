class TripleExtractor:
    def __init__(self,
        annotators = ['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref', 'openie'],
        properties={
            "openie.triple.strict":"true",
            "openie.resolve_coref" :"true"
        },
        memory_use = '8G',
        server_endpoint = 'http://localhost:9000',
        timeout = 1.5 * 60 * 1000,
        verbose = False,
        STOP_WORDS = ["O", "DATE", "IDEOLOGY", "NUMBER", "DURATION", "MONEY", "CURRENCY", "PERCENT", "MISC"]
    ):
        import os
        self.corenlp_dir = './corenlp'
        os.environ["CORENLP_HOME"] = self.corenlp_dir

        self.annotators = annotators
        self.properties = properties
        self.memory_use = memory_use
        self.server_endpoint = server_endpoint
        self.timeout = timeout
        self.be_quiet = not verbose
        self.STOP_WORDS = STOP_WORDS
        self.RELATION_MATRIX = []
        self.doc = None
        self.triples = None
    
    def install(self):
        import subprocess
        import sys
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "stanza"]
        )
        import stanza
        stanza.install_corenlp(dir=self.corenlp_dir)
    
    # BASIC STANZA 
    def get_doc(self, text):
        from stanza.server import CoreNLPClient
        with CoreNLPClient(
            annotators=self.annotators,
            properties=self.properties,
            memory=self.memory_use, 
            endpoint=self.server_endpoint, 
            be_quiet=self.be_quiet,
            timeout=self.timeout
        ) as client:
            self.doc = client.annotate(text)
    
    # NER RELATED
    def getNER(self, tokens):
        ner = []
        for pos in tokens:
            ner.append(self.doc.sentence[pos.sentenceIndex].token[pos.tokenIndex].ner)
        return ner
    
    def GetAllNER(self, triple):
        all_NER = []
        if(hasattr(triple, "subjectTokens")):
            all_NER += self.getNER(triple.subjectTokens)

        if(hasattr(triple, "relationTokens")):
            all_NER += self.getNER(triple.relationTokens)

        if(hasattr(triple, "objectTokens")):
            all_NER += self.getNER(triple.objectTokens)
        all_NER += [triple.subject, triple.relation, triple.object]
        return all_NER
    
    def isValidTokens(self, tokens):
        are_any_tokens_in_stop_words = any(item in self.getNER(tokens) for item in self.STOP_WORDS)
        return not are_any_tokens_in_stop_words

    def isValidTriple(self, triple):
        self_loop = triple.subject == triple.object
        isValidSubject = self.isValidTokens(triple.subjectTokens)
        isValidObject = self.isValidTokens(triple.objectTokens)
        return isValidSubject and isValidObject and not self_loop

    def getValidTriples(self):
        all_triples = []
        for sentence in self.doc.sentence:
            for triple in getattr(sentence, "openieTriple"):
                if self.isValidTriple(triple):
                    all_triples.append(triple)
        self.triples = all_triples
    
    # Relation Related
    def import_FB15k_relations(self):
        import pandas as pd
        relations = pd.read_csv("relations.txt", sep="\t", names=['relation', 'index'])
        relations = relations['relation'].values
        RELATION_MATRIX = []

        for rel in relations:
            RELATION_MATRIX.append(rel.split("/")[1:])
        self.RELATION_MATRIX = RELATION_MATRIX       

    def getNewRelation(self, triple):
        import numpy as np
        curr_ner = self.GetAllNER(triple)
        curr_ner = [n.lower() for n in curr_ner]
        most_common = []
        most_common_score = 0
        for rel in self.RELATION_MATRIX:
            common_score = sum(el in np.unique(curr_ner) for el in np.unique(rel))
            if common_score == most_common_score:
                most_common.append(rel)
            if common_score > most_common_score:
                most_common = []
                most_common.append(rel)
                most_common_score = common_score
        if(most_common_score > 1):
            return "/".join(most_common[0])
        return False

    def set_experimental_relationship(self):
        new_triples = self.triples
        for triple in new_triples:
            new_rel = self.getNewRelation(triple)
            if(new_rel):
                triple.relation = new_rel
        self.triples = new_triples
    
    def set_only_FB15K_valid_triples(self):
        new_triples = []
        for triple in self.triples:
            if triple.relation.__contains__("/"):
                new_triples.append(triple)
        self.triples = new_triples
        
    def get_pref_ner_index(self, ners):
        import numpy as np
        if "PERSON" in ners:
            return ners == "PERSON"
        if "ORGANIZATION" in ners:
            return ners == "ORGANIZATION"
        if "LOCATION" in ners:
            return ners == "LOCATION"
        return np.array([1]*len(ners))

    def get_pref_entity(self, triple):
        import numpy as np
        obj_ners = self.getNER(triple.objectTokens)
        if not obj_ners.count(obj_ners[0])==len(obj_ners):
            new_obj = np.array(triple.object.split(" "))[self.get_pref_ner_index(np.array(obj_ners))]
            new_obj = " ".join(new_obj)
            triple.object = new_obj
            
        sub_ners = self.getNER(triple.subjectTokens)
        if not sub_ners.count(sub_ners[0])==len(sub_ners):
            new_sub = np.array(triple.subject.split(" "))[self.get_pref_ner_index(np.array(sub_ners))]
            new_sub = " ".join(new_sub)
            triple.subject = new_sub

    def set_prefered_ner(self):
        for triple in self.triples:
            self.get_pref_entity(triple)
            
    def print_triples(self):
        print("{:25s}\t{:60s}\t{:20s}".format("Head","Relation","Tail"))
        for triple in self.triples:
            s = triple.subject
            r = triple.relation
            o = triple.object
            print("{:25s}\t{:60s}\t{:20s}".format(s,r,o))