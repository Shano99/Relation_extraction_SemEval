__author__ = "David S. Batista"
__email__ = "dsbatista@gmail.com"

import fileinput
import re
from typing import Any, Optional, Set

from gensim.models import KeyedVectors
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

from breds.reverb import Reverb
from breds.seed import Seed
import random

class Config:  # pylint: disable=too-many-instance-attributes, too-many-arguments
    """
    Initializes a configuration object with the parameters from the config file:
    - Reads the word2vec model.
    - Initializes the lemmatizer and the stopwords list.
    - Set the weights for the unknown and negative instances.
    - Set the POS tags to be filtered out.
        http://www.ling.upenn.edu/courses/Fall_2007/ling001/penn_treebank_pos.html
        select everything except stopwords, ADJ and ADV
    - Set the regex to clean the text.
    - Set the threshold for the similarity between the patterns and the instances.
    - Set the threshold for the confidence of the patterns.
    - Initialize the Reverb object.
    """

    def __init__(
        self,
        word2vec_model_path: str,
        similarity: float,
        confidence: float,
        number_iterations: int,
        alpha : float,
        beta :  float,
        gamma : float
        
    ) -> None:  # noqa: C901
        
        self.context_window_size: int = 4
        self.min_tokens_away: int = 0
        self.max_tokens_away: int = 20
        self.similarity: float = 0.6
        self.alpha: float = alpha
        self.beta: float = beta
        self.gamma: float = gamma
        self.min_pattern_support: int = 4
        self.w_neg: float = 2
        self.w_unk: float = 0.0
        self.w_updt: float = 0.5


        self.filter_pos = ["JJ", "JJR", "JJS", "RB", "RBR", "RBS", "WRB"]
        self.stopwords = stopwords.words("english")
        self.lemmatizer = WordNetLemmatizer()
        self.regex_clean_simple = re.compile("</?[A-Z]+>", re.U)
        self.tags_regex = re.compile("</?[A-Z]+>", re.U)
        self.positive_seed_tuples: Set[Any] = set()
        self.negative_seed_tuples: Set[Any] = set()
        self.e1_type: str
        self.e2_type: str
        self.threshold_similarity = similarity
        self.instance_confidence = confidence
        self.number_iterations = number_iterations
        self.word2vec_model_path = word2vec_model_path
        self.reverb = Reverb()
        self.word2vec: Any
        self.vec_dim: int

    def print_config(self) -> None:  # pragma: no cover
        # pylint: disable=expression-not-assigned
        """
        Prints the configuration parameters.
        """
        print("Configuration parameters")
        print("========================\n")
        # print("e1 type              :", self.e1_type)
        # print("e2 type              :", self.e2_type)
        print("context window       :", self.context_window_size)
        print("max tokens away      :", self.max_tokens_away)
        print("min tokens away      :", self.min_tokens_away)
        print("word2vec model       :", self.word2vec_model_path)
        print("\n")
        print("alpha                :", self.alpha)
        print("beta                 :", self.beta)
        print("gamma                :", self.gamma)
        print("\n")

        print("threshold_similarity :", self.threshold_similarity)
        print("instance confidence  :", self.instance_confidence)
        print("min_pattern_support  :", self.min_pattern_support)
        print("iterations           :", self.number_iterations)
        # print("iteration wUpdt      :", self.w_updt)
        print("\n")

    def read_word2vec(self, path: str) -> KeyedVectors:  # type: ignore
        """Reads the word2vec model."""

        print("Loading word2vec model ...\n")
        word2vec = KeyedVectors.load_word2vec_format(path, binary=True)
        self.vec_dim = word2vec.vector_size
        print("The word2vec mode has ",self.vec_dim, "dimensions")
        return word2vec
    """
    Code ENHANCEMENT BEGIN: ---------------------------------------------------------------------------------------------------------------------------
    Description: Randomly choose a set of seeds as positve seeds for the bootstrapping process.
    """   
    def randomize_seeds(self, key: str, seeds_obj, holder: Set[Any], percentage ):
        self.e1_type,self.e2_type = key.split("_")
        self.e1_type.upper()
        self.e2_type.upper()
        seed_count = len(seeds_obj) // percentage  # take given percentage of seeds as positive seeds 
        if seed_count == 0:
            seed_count=1
        selected_items = random.sample(seeds_obj, seed_count) # randomize the sample
        for seed in selected_items:
           ent1,ent2 = seed.split(";")
           seed = Seed(ent1, ent2)
           holder.add(seed)
        
    """
    Code ENHANCEMENT END: ----------------------------------------------------------------------------------------------------------------------------
    """