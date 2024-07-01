import re,random
import json
import operator
import os
import pickle
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from gensim import matutils
from nltk.data import load
from numpy import dot
from tqdm import tqdm

from breds.breds_tuple import BREDSTuple
from breds.commons import blocks
from breds.config import Config
from breds.pattern import Pattern
from breds.seed import Seed
from breds.sentence import Sentence

# Download the POS Tagger
import nltk
# nltk.download('maxent_treebank_pos_tagger')

class BREDS:
    """
    BREDS is a system that extracts relationships between named entities from text.
    """
     # Function to download a package only if it's not already downloaded
    def download_nltk_data(self,package):
        try:
            nltk.data.find(package.split('/')[1])
        except LookupError:
            print(" downloading package")
            nltk.download(package.split('/')[1])

    def __init__(
        self,
        word2vec_model_path: str,
    ):
        # pylint: disable=too-many-arguments
        self.curr_iteration = 0
        self.config = Config(
            word2vec_model_path, 0.3, 0.3, 0, 0, 1.0, 0
        )
        self.config.word2vec = self.config.read_word2vec(self.config.word2vec_model_path) # Load embedding model

        
        self.total_entities = {}
        self.total_valid_entities = {}   

        self.TP = {} # True Positive
        self.FP = {} # False Positive
        self.FN = {} # False Negative

        # Metrics for micro-averaging
        self.total_TP = 0 
        self.total_FP = 0
        self.total_FN = 0
        self.model = {} 

        
        
   
    
    def test_tuple(self,tpl):
        """
        Compares each tuple with the pattern and calculates the similarity value. 
        Finally, it takes the pattern with the highest similarity value as the relation type.
        """
        sim_best: float = 0.0
        pattern_best = None
        key_best = None
        for key in self.model: # Read patterns from the trained model
            for pattern in self.model[key]:
                accept, score = self.similarity_all(tpl, pattern)
                if accept is True: 
                    if score > sim_best: # Check for highest similarity score
                        sim_best = score
                        pattern_best = pattern
                        key_best = key
                        
        if key_best is None: # If no relation is matched, assign it to Other relation.
            key_best="OTHER_OTHER"

        return key_best
     
    def init_testing(self) -> None:
        """
        Generate tuples instances from a text file with sentences where named entities are already tagged.
        """
        
        sentences_file = "./data/sentence_test"
        # Use the function to ensure necessary packages are downloaded
        download_nltk_data('corpora/stopwords')
        download_nltk_data('taggers/maxent_treebank_pos_tagger')
        tagger = load("taggers/maxent_treebank_pos_tagger/english.pickle")
        
        print("Testing input sentences")
        with open(sentences_file, "r") as file:
            data = json.load(file)
            
            for key in data.keys(): # initialise with 0
                self.TP[key]=0
                self.FP[key]=0
                self.FN[key]=0
                
            for true_label in data:
                self.total_entities[true_label]=len(data[true_label])
                self.processed_tuples: List[BREDSTuple] = []
                
                for line in data[true_label]:
                    self.config.e1_type,self.config.e2_type = true_label.split("_")
                    sentence = Sentence(
                        line,
                        self.config.e1_type.upper(),
                        self.config.e2_type.upper(),
                        self.config.max_tokens_away,
                        self.config.min_tokens_away,
                        self.config.context_window_size,
                        tagger,
                    )
                    
                    for rel in sentence.relationships:
                        tpl = BREDSTuple(
                            rel.ent1, rel.ent2, rel.sentence, rel.before, rel.between, rel.after, self.config
                        )
                    
                        self.processed_tuples.append(tpl) 
                        pred=self.test_tuple(tpl)
                        
                        if true_label==pred: # Annotated-Correct
                            self.TP[true_label]+=1
                            self.total_TP+=1
                        else:
                            # Annotated - Not Correct
                            if pred!='OTHER_OTHER': # Assuming 'Other' is considered as a negative prediction
                                self.FP[pred]+=1
                                self.total_FP+=1
                                
                            # Not Annotated - Correct
                            if true_label!='OTHER_OTHER': # Assuming 'Other' as negative label
                                self.FN[true_label]+=1
                                self.total_FN+=1


                self.total_valid_entities[true_label.upper()]=len(self.processed_tuples)
                
            
    def similarity_3_contexts(self, tpl: BREDSTuple, pattern: BREDSTuple) -> float:
        """
        Calculates the cosine similarity between the context vectors of a pattern and a tuple.
        """
        (bef, bet, aft) = (0, 0, 0)

        if tpl.bef_vector is not None and pattern.bef_vector is not None:
            bef = dot(matutils.unitvec(tpl.bef_vector), matutils.unitvec(pattern.bef_vector))

        if tpl.bet_vector is not None and pattern.bet_vector is not None:
            bet = dot(matutils.unitvec(tpl.bet_vector), matutils.unitvec(pattern.bet_vector))

        if tpl.aft_vector is not None and pattern.aft_vector is not None:
            aft = dot(matutils.unitvec(tpl.aft_vector), matutils.unitvec(pattern.aft_vector))

        return self.config.alpha * bef + self.config.beta * bet + self.config.gamma * aft

    def similarity_all(self, tpl: BREDSTuple, extraction_pattern: Pattern) -> Tuple[bool, float]:
        """
        Calculates the cosine similarity between all patterns part of a cluster (i.e., extraction pattern) and the
        vector of a ReVerb pattern extracted from a sentence.

        Returns the max similarity score
        """
        good: int = 0
        bad: int = 0
        max_similarity: float = 0.0

        for pattern in list(extraction_pattern.tuples):
            score = self.similarity_3_contexts(tpl, pattern)
            if score > max_similarity:
                max_similarity = score
            if score >= self.config.threshold_similarity:
                good += 1
            else:
                bad += 1

        if good >= bad:
            return True, max_similarity

        return False, 0.0
    """
    Code CHANGE BEGIN: --------------------------------------------------------------------------------------------------------------------------------
    """
    def read_test_file(self,test_file): # create all_sentence and all_seeds_file:
        
        sent_pattern = r'^\d+\s'         # Regex pattern to extract sentence from input file
        e1_pattern   = r'<e1>(.*?)</e1>' # Regex pattern to extract e1 entity from input file
        e2_pattern   = r'<e2>(.*?)</e2>' # Regex pattern to extract e2 entity from input file
        
        sentence_set={}

        print("Reading the ",test_file,"...")
        
        with open(test_file, 'r') as input: # Reads the input test file 
            sentence = input.readline().strip()
            while sentence:
                relation = input.readline().strip()
                comment = input.readline()
                blank = input.readline()
                
                if relation[-7:] == "(e1,e2)": # Extracts relation types e1 and e2 eg: (CAUSE,EFFECT)
                    e1 = relation[:relation.find('-')]
                    e2 = relation[relation.find('-')+1:relation.find('(')]
                    
                    
                elif relation[-7:] == "(e2,e1)":  # Extracts reverse relation types e1 and e2 eg: (EFFECT,CAUSE)
                    e2 = relation[:relation.find('-')]
                    e1 = relation[relation.find('-')+1:relation.find('(')]
                    
                else:
                    e1 = "OTHER"
                    e2 = "OTHER"
                 
                key = (e1+"_"+e2).upper()
               
                e1_entity = re.findall(e1_pattern, sentence)[0]
                e2_entity = re.findall(e2_pattern, sentence)[0]

                """
                Extracts sentences and replaces e1,e2 with respective relation types.
                E.g., "Diet fizzy <CAUSE>drinks</CAUSE> and meat cause heart disease and <EFFECT>diabetes</EFFECT>.\"
                """
                
                result_string = re.sub(sent_pattern, '', sentence)
                res = result_string.replace("<e1>", "<"+e1.upper()+">")
                res = res.replace("</e1>", "</"+e1.upper()+">")
                res = res.replace("<e2>", "<"+e2.upper()+">")
                res = res.replace("</e2>", "</"+e2.upper()+">")
               
                if key in sentence_set:
                    sentence_set[key].append(res)
                else:
                    sentence_set[key]=[res]
                sentence = input.readline().strip()

        print("Writing extracted sentences to sentences_test file...")
        
        with open("./data/sentence_test","w") as json_file:  # Write the sentences to sentences_test file
            json.dump(sentence_set, json_file, indent=4)


    def load_model(self,model_file): # Load the model from disk
        with open(model_file, "rb") as f_in:
            print("\nLoading model from disk...")
            self.model = pickle.load(f_in)
        

    def test_user_input(self,sentence):
        tagger = load("taggers/maxent_treebank_pos_tagger/english.pickle")
        self.config.e1_type = "e1"
        self.config.e2_type = "e2"

        # check if the entered sentence is in the right format
        sentence = Sentence(
                        sentence.strip(),
                        self.config.e1_type,
                        self.config.e2_type,
                        self.config.max_tokens_away,
                        self.config.min_tokens_away,
                        self.config.context_window_size,
                        tagger,
                    )
    
        for rel in sentence.relationships:
            tpl = BREDSTuple(
                rel.ent1, rel.ent2, rel.sentence, rel.before, rel.between, rel.after, self.config
            )
            key_best=self.test_tuple(tpl)
            
            if key_best:
                print("The relation extracted from the sentence is : " + key_best)
            else:
                print("No relations extracted")
            return
        print("The input sentence is not of the following format:\n")
        print("1. It must contain two entities enclosed in <e1>...</e1> and <e2>...</e2> tags.\n2. It must contain atleast one valid word between the two entities.\nPS: Valid words includes all words excluding blankspace, stop words and punctutations.\nNote: The major weakness of this method is that it cannot extract relations from sentences where there is no BETWEEN context.")
        print("STOPWORDS: ",stopwords,"\n")
    def evaluate_test_set(self): # micro-averaging to find Precision, Recall and F1_score
     
        precision = {}
        recall = {}
        F1_score = {}

        print("Relation\t\tPrecision\t\tRecall\t\tF1\n")
        
        for key in self.total_entities:
            
            try:
                precision[key]=self.TP[key]/(self.TP[key]+self.FP[key])
            except ZeroDivisionError:
                precision[key]=0
                
            try:
                recall[key]=self.TP[key]/(self.TP[key]+self.FN[key])
            except ZeroDivisionError:
                recall[key]=0
                
            try:
                F1_score[key]=2*precision[key]*recall[key]/(precision[key]+recall[key])
            except ZeroDivisionError:
                F1_score[key]=0
                
            print(key,"\t",precision[key],"\t",recall[key],"\t",F1_score[key],"\n")

        total_precision=self.total_FP/(self.total_TP+self.total_FP)
        total_recall=self.total_TP/(self.total_TP+self.total_FN)
        total_F1_score=2*total_precision*total_recall/(total_precision+total_recall)
        
        print("Total\t",total_precision,"\t",total_recall,"\t",total_F1_score,"\n")
        # print("Total number of test sentences: ",self.total_entities,"\n")
        # print("Total number of valid sentences: ",self.total_valid_entities,"\n")

        
        with open("Fine_tuning_values.txt","a") as f_out:
            f_out.write("Test set evaluation:\n")
            f_out.write("Precision\t\tRecall\t\tF1\n")
            f_out.write(str(total_precision)+"\t"+str(total_recall)+"\t"+str(total_F1_score)+"\n\n")
                  
    """
    Code CHANGE END: ------------------------------------------------------------------------------------------------------------------------------
    """
        