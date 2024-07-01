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

PRINT_TUPLES = False
PRINT_PATTERNS = False

# Download the POS Tagger
import nltk
nltk.download('maxent_treebank_pos_tagger')

class BREDS:
    """
    BREDS is a system that extracts relationships between named entities from text.
    """

    def __init__(
        self,
        word2vec_model_path: str,
        similarity: float,
        confidence: float,
        number_iterations: int,
        alpha :  float,
        beta : float,
        gamma : float
    ):
        self.curr_iteration = 0
        self.config = Config(
            word2vec_model_path,similarity, confidence, number_iterations, alpha, beta, gamma
        )
        self.config.word2vec = self.config.read_word2vec(self.config.word2vec_model_path) # reads the word2vec embedding model
        self.config.print_config() # prints the current configurations

        """
        Code Change BEGIN: -----------------------------------------------------------------------------------------------------------------------------
        Description: Initializing below variables for calculating the training accuracy.
        """
        self.total_entities = {}        # Total number of sentences for each relation type
        self.total_valid_entities = {}  # Total number of valid sentences (excluding the ones without BET context) for each relation type
        self.extracted_entities = {}    # Total number of relations extracted for each relation
        self.total_train_accuracy = {}  # Training Accuracy w.r.t all the sentences for each relation type
        self.valid_train_accuracy = {}  # Training Accuracy w.r.t valid sentences for each relation type
        self.model = {}                 # Final set of extracted patterns for each relation type
        
        """
        Code Change END: -------------------------------------------------------------------------------------------------------------------------------
        """
         
    def read_train_file(self,train_file): 
        """
        Code Change BEGIN: ---------------------------------------------------------------------------------------------------------------------------- 
        Description : Reads the input TRAIN_FILE.txt, extracts the sentences and relations and creates the all_sentences_train and all_seeds_train                       json files. These files are stored in ./data directory.
        """
        
        sent_pattern = r'^\d+\s'         # Regex pattern to extract sentence from input file
        e1_pattern   = r'<e1>(.*?)</e1>' # Regex pattern to extract e1 entity from input file
        e2_pattern   = r'<e2>(.*?)</e2>' # Regex pattern to extract e2 entity from input file
        
        sentence_set = {}
        
        print("Reading the ",train_file,"...")
        
        with open(train_file, 'r') as input: # Reads the input train file 
            sentence = input.readline().strip()
            while sentence:
                relation = input.readline().strip()
                comment = input.readline()
                blank = input.readline()
        
                if relation[-7:] == "(e1,e2)":  # Extracts relation types e1 and e2 eg: (CAUSE,EFFECT)
                    e1 = relation[:relation.find('-')]
                    e2 = relation[relation.find('-')+1:relation.find('(')]
                    
                if relation[-7:] == "(e2,e1)": # Extracts reverse relation types e1 and e2 eg: (EFFECT,CAUSE)
                    e2 = relation[:relation.find('-')]
                    e1 = relation[relation.find('-')+1:relation.find('(')]
                    
                # Exclude Sentences with relation type - Other
                 
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
                    
        print("Writing extracted sentences to sentences_train file...")
        
        with open("./data/sentence_train","w") as json_file: # Write the sentences to sentences_train file
            json.dump(sentence_set, json_file, indent=4)

        """
        Code Change END: -------------------------------------------------------------------------------------------------------------------------------
        """
        
    def generate_tuples(self) -> None:
        
        """
        Generate tuples instances from a input text file with sentences where named entities are already tagged.

        """
        
        seed_set={} 
        sentences_file = "./data/sentence_train" 
        tagger = load("taggers/maxent_treebank_pos_tagger/english.pickle") # Load the POS-tagger

        print("\nProcessing input sentences")
        with open(sentences_file, "r") as file:
            data = json.load(file)

            for key in data:
                self.total_entities[key] = len(data[key])
                self.processed_tuples: List[BREDSTuple] = []
                
                for line in data[key]:
                    self.config.e1_type,self.config.e2_type = key.split("_")
                    
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

                        """
                        Code Change:
                        Extract the relation sets from the sentences. Example, (CAUSE,EFFECT) - (drinks,effects)
                        """
                        if key in seed_set:
                            seed_set[key].append(rel.ent1+";"+rel.ent2)
                        else:
                            seed_set[key]=[rel.ent1+";"+rel.ent2]
                        

                # print("\n", len(self.processed_tuples), "tuples generated for ",key)
                self.total_valid_entities[key.upper()]=len(self.processed_tuples) 
                # print(key,self.total_valid_entities[key.upper()])
                
                # print("Writing generated tuples to disk")
                with open("./preprocessed_files/"+key+"_processed_tuples.pkl", "wb") as f_out:
                    pickle.dump(self.processed_tuples, f_out)
                    
            with open("./data/seeds_file", "w") as json_file: # Write all the seeds(tuples) to all_seeds file
                json.dump(seed_set, json_file, indent=4)
            
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

    def match_seeds_tuples(self) -> Tuple[Dict[Tuple[str, str], int], List[BREDSTuple]]:
        """
        Checks if the extracted tuples match the seeds tuples.
        """
        matched_tuples: List[BREDSTuple] = []
        count_matches: Dict[Tuple[str, str], int] = defaultdict(int)
        for tpl in self.processed_tuples:
            for sent in self.config.positive_seed_tuples:
                if tpl.ent1 == sent.ent1 and tpl.ent2 == sent.ent2:
                    matched_tuples.append(tpl)
                    count_matches[(tpl.ent1, tpl.ent2)] += 1

        return count_matches, matched_tuples

    def write_relationships_to_disk(self,key : str) -> None:
        """
        Writes the extracted relationships to disk.
        The output file is a JSONL file with one relationship per line.
        """
        
        print("\nWriting extracted relationships to disk")
        with open("./Relations_extracted/"+key+"_relationships.jsonl", "wt", encoding="utf8") as f_out:
            for tpl in sorted(list(self.candidate_tuples.keys()), reverse=True):
                f_out.write(json.dumps(tpl.to_json()) + "\n")
        
    def cluster_tuples(self, matched_tuples: List[BREDSTuple]) -> None:
        """
        Single Pass Clustering Algorithm
        Cluster the matched tuples to generate patterns
        """
        # Initialize: if no patterns exist, first tuple goes to first cluster
        if len(self.patterns) == 0:
            self.patterns.append(Pattern(matched_tuples[0]))

        for tpl in tqdm(matched_tuples):
            max_similarity: float = 0.0
            max_similarity_cluster_index = 0

            # go through all patterns(clusters of tuples) and find the one with the highest similarity score
            for i in range(0, len(self.patterns), 1):
                extraction_pattern = self.patterns[i]
                accept, score = self.similarity_all(tpl, extraction_pattern)

                if accept is True and score > max_similarity:
                    max_similarity = score
                    max_similarity_cluster_index = i

            # if max_similarity < min_degree_match create a new cluster having this tuple as the centroid
            if max_similarity < self.config.threshold_similarity:
                cluster = Pattern(tpl)
                self.patterns.append(cluster)

            # if max_similarity >= min_degree_match add to the cluster with the highest similarity
            else:
                self.patterns[max_similarity_cluster_index].add_tuple(tpl)

    def debug_patterns_1(self) -> None:
        """
        Prints the patterns to the console
        """
        count = 1
        print("\nPatterns:")
        for pattern in self.patterns:
            print(count)
            for pattern_tuple in pattern.tuples:
                print("BEF", pattern_tuple.bef_words)
                print("BET", pattern_tuple.bet_words)
                print("AFT", pattern_tuple.aft_words)
                print("========")
                print("\n")
            count += 1

    def debug_patterns_2(self) -> None:
        """
        Prints the patterns to the console
        """
        print("\nPatterns:")
        for pattern in self.patterns:
            for tpl in pattern.tuples:
                print("BEF", tpl.bef_words)
                print("BET", tpl.bet_words)
                print("AFT", tpl.aft_words)
                print("========")
            print("Positive", pattern.positive)
            print("Negative", pattern.negative)
            print("Unknown", pattern.unknown)
            print("Tuples", len(pattern.tuples))
            print("Pattern Confidence", pattern.confidence)
            print("\n")

    def debug_tuples(self) -> None:
        """
        Prints the tuples to the console
        """
        if PRINT_TUPLES is True:
            extracted_tuples = list(self.candidate_tuples.keys())
            tuples_sorted = sorted(extracted_tuples, key=lambda tpl: tpl.confidence, reverse=True)
            for tpl in tuples_sorted:
                print(tpl.sentence)
                print(tpl.ent1, tpl.ent2)
                print(tpl.confidence)
                print("\n")

    def updated_tuple_confidence(self) -> None:
        """
        Updates the confidence of the tuples
        """
        print("\n\nCalculating tuples confidence")
        for tpl, patterns in self.candidate_tuples.items():
            confidence: float = 1.0
            tpl.confidence_old = tpl.confidence
            for pattern in patterns:
                confidence *= 1 - (pattern[0].confidence * pattern[1])
            tpl.confidence = 1 - confidence

    def generate_candidate_tuples(self) -> None:
        """
        Generates the candidate tuples
        """
        for tpl in tqdm(self.processed_tuples):
            sim_best: float = 0.0
            for extraction_pattern in self.patterns:
                accept, score = self.similarity_all(tpl, extraction_pattern)
                if accept is True:
                    extraction_pattern.update_selectivity(tpl, self.config)
                    if score > sim_best:
                        sim_best = score
                        pattern_best = extraction_pattern

            if sim_best >= self.config.threshold_similarity:
                # if this tuple was already extracted, check if this
                # extraction pattern is already associated with it,
                # if not, associate this pattern with it and store the
                # similarity score
                patterns = self.candidate_tuples[tpl]
                if patterns is not None:
                    if pattern_best not in [x[0] for x in patterns]:
                        self.candidate_tuples[tpl].append((pattern_best, sim_best))

                # If this tuple was not extracted before
                # associate this pattern with the instance
                # and the similarity score
                else:
                    self.candidate_tuples[tpl].append((pattern_best, sim_best))

    def init_bootstrap(self) -> None:  # noqa: C901
        """Initializes the bootstrap process"""

        directory_path = "./preprocessed_files"
        all_seeds_file = "./data/seeds_file"
        with open(all_seeds_file,'r') as seeds_file:
            all_seeds = json.load(seeds_file)
        """
        Code ENHANCEMENT BEGIN: -----------------------------------------------------------------------------------------------------------------------
        Description: If no patterns are generated, re-populate the seeds randomly and execute the bootstrap process again upto 5 trials.
        """
        for key in all_seeds:
            flag = True # Set Flag to mark 0 patterns are generated 
            trials = 1  # Begin the trial
        
            while(flag and trials<=5): # Run while flag is set and upto 5 trials
                self.config.positive_seed_tuples: Set[Any] = set()
                percentage = 10   # Extract 10% of total seeds of each relation as postive_seeds.
                self.config.randomize_seeds(key, all_seeds[key], self.config.positive_seed_tuples,percentage) # Randomly choose the postive seeds
                
                sentence_file_path = os.path.join(directory_path, key+"_processed_tuples.pkl") 
                """
                Code ENHANCEMENT END: -----------------------------------------------------------------------------------------------------------------
                """
                # Read each processed.pkl file
                if os.path.isfile(sentence_file_path):
                    # Process the file here
                    
                    self.processed_tuples: List[BREDSTuple] = []
                    
                    with open(sentence_file_path, "rb") as f_in:
                        self.processed_tuples = pickle.load(f_in)
                        
                    self.curr_iteration = 0
                    self.patterns: List[Pattern] = []
                    self.candidate_tuples: Dict[BREDSTuple, List[Tuple[Pattern, float]]] = defaultdict(list)
            
                    while self.curr_iteration <= self.config.number_iterations:
                        print("==========================================")
                        print("\nStarting iteration", self.curr_iteration)
                        
                        # print("\nLooking for seed matches of:")
                        # for sent in self.config.positive_seed_tuples: 
                        #     print(f"{sent.ent1}\t{sent.ent2}")
            
                        # Looks for sentences matching the seed instances
                        count_matches, matched_tuples = self.match_seeds_tuples()
            
                        if len(matched_tuples) == 0: # Exit, if no matches are found
                            print("\nNo seed matches found")
                            print("---------------------------------")
                            break
                            
            
                        else:
                            # print("\nNumber of seed matches found")
                            # for seed_match in sorted(list(count_matches.items()), key=operator.itemgetter(1), reverse=True):
                            #     print(f"{seed_match[0][0]}\t{seed_match[0][1]}\t{seed_match[1]}")
                            
                            print(f"\n{len(matched_tuples)} tuples matched")
            
                            # Cluster the matched instances, to generate patterns/update patterns
                            print("\nClustering matched instances to generate patterns")
                            self.cluster_tuples(matched_tuples)
            
                            # Eliminate patterns supported by less than 'min_pattern_support' tuples
                            new_patterns = [p for p in self.patterns if len(p.tuples) >= self.config.min_pattern_support]
                            self.patterns = new_patterns
                            
                            print(f"\n{len(self.patterns)} patterns generated")
                            if PRINT_PATTERNS is True:
                                self.debug_patterns_1()
            
                            if self.curr_iteration == 0 and len(self.patterns) == 0: # Exit when no patterns are generated
                                print("No patterns generated")
                                print("---------------------------------")
                                trials+=1
                                break
            
                            # Look for sentences with occurrence of seeds semantic types (e.g., CAUSE - EFFECT)
                            # This was already collect, and it's stored in: self.processed_tuples
                            #
                            # Measure the similarity of each occurrence with each extraction pattern and store each pattern that
                            # has a similarity higher than a given threshold
                            #
                            # Each candidate tuple will then have a number of patterns that extracted it each with an associated
                            # degree of match.
                            print("Number of tuples to be analyzed:", len(self.processed_tuples))
                            print("\nCollecting instances based on extraction patterns")
                            self.generate_candidate_tuples()
            
                            # update all patterns confidence
                            for pattern in self.patterns:
                                pattern.update_confidence(self.config)
            
                            if PRINT_PATTERNS is True:
                                self.debug_patterns_2()
            
                            # update tuple confidence based on patterns confidence
                            self.updated_tuple_confidence()
            
                            # sort tuples by confidence and print
                            self.debug_tuples()
            
                            print(f"Adding tuples to seed with confidence >= {str(self.config.instance_confidence)}")
                            for tpl, _ in self.candidate_tuples.items():
                                if tpl.confidence >= self.config.instance_confidence:
                                    seed = Seed(tpl.ent1, tpl.ent2)
                                    self.config.positive_seed_tuples.add(seed)
            
                            # increment the number of iterations
                            self.curr_iteration += 1
                            flag = False  # Reset flag implying that some patterns are generated
                            
            
                    self.write_relationships_to_disk(key)
                    """
                    Code CHANGE BEGIN: ---------------------------------------------------------------------------------------------------------------
                    Description: Calculate the training accuracy (number of relations extracted / total number of relations)
                    """
                    self.extracted_entities[key] = len(self.candidate_tuples)
                    
                    # Total train accuracy is calculated upon all the sentences(including the invalid ones)
                    # Invalid sentences are the ones without any BETWEEN context and which includes stop_words
                    self.total_train_accuracy[key]=(self.extracted_entities[key]/self.total_entities[key])*100
    
                    # Total valid train accuracy is calculated upon all the valid sentences(excluding the invalid ones)
                    self.valid_train_accuracy[key]=(self.extracted_entities[key]/self.total_valid_entities[key])*100
    
                    # store patterns in model object
                    self.model[key]=self.patterns

        # save model which contains patterns as model.pkl file
        print("Saving extracted patterns as model to disk")
        with open("model.pkl", "wb") as f_out:
            pickle.dump(self.model, f_out)
        """
        Code CHANGE END: ------------------------------------------------------------------------------------------------------------------------------
        """

    """
    Code CHANGE BEGIN: --------------------------------------------------------------------------------------------------------------------------------
    Description: Print the training accuracy (number of relations extracted / total number of relations)
    """
    def training_accuracy(self):
        
        total_sent = 0
        valid_sent = 0
        extracted_sent = 0
        print("Calculating training accuracy and saving to disk as Training_accuracy.txt")
        with open("Fine_tuning_values.txt","a") as f_out: # Save the calculated accuracies in a file
            f_out.write("Hyperparameters:\n")
            f_out.write("Similarity:"+str(self.config.threshold_similarity)+"\n")
            f_out.write("Confidence:"+str(self.config.instance_confidence)+"\n")
            f_out.write("Alpha:"+str(self.config.alpha)+"\n")
            f_out.write("Beta:"+str(self.config.beta)+"\n")
            f_out.write("Gamma:"+str(self.config.gamma)+"\n")
            f_out.write("Number of Iterations:"+str(self.config.number_iterations)+"\n\n")

            
            f_out.write("Relation\t   Total Accuracy for each relation\tValid Accuracy for each relation\n")
            for key in self.total_train_accuracy: # accuracy for each relation
                total_sent += self.total_entities[key]
                valid_sent += self.total_valid_entities[key]
                extracted_sent += self.extracted_entities[key]
                
                f_out.write(key+"\t\t"+str(self.total_train_accuracy[key])+"\t\t"+str(self.valid_train_accuracy[key])+"\n")

            # total training accuracy for the model
            f_out.write("Total\t\t\t"+str((extracted_sent/total_sent)*100)+"\t\t\t"+str((extracted_sent/valid_sent)*100)+"\n\n") 
            
            f_out.write("Total number of sentences (excluding OTHER relation): "+str(total_sent)+"\n" )
            f_out.write("Total number of valid sentences: "+str(valid_sent)+"\n"  )
            f_out.write("Total number of tuples extracted: "+str(extracted_sent)+"\n\n\n"  )
    """
    Code CHANGE END: ------------------------------------------------------------------------------------------------------------------------------
    """
        