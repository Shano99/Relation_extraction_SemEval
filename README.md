# BOOTSTRAPPING - NLP

## Description

BREDS extracts relationships using a bootstrapping/semi-supervised approach, it relies on an initial set of seeds, i.e. pairs of named-entities representing relationship type to be extracted.  

The algorithm expands the initial set of seeds using distributional semantics to generalize the relationship while 
limiting the semantic drift.


## Installation

To install the required packages, please run the following command:

```bash
pip install -r requirements.txt
```
This will install all the necessary dependencies for running the project.

## Usage

### Training

To train the model, use the following command:

```bash
python train.py --train_file "TRAIN_FILE.txt"
```

### Testing

To test the trained model, you can run the following command:

```bash
python test_model.py --test_file "TEST_FILE.txt"
```


### Testing User Input

To test the model with user input, you can run the following command:

```bash
python test_user_input.py --sentence "<e1>Acne</e1> is caused by <e2>stress</e2>."
```

The input sentence must be of the following format:
1. It must contain two entities enclosed in <e1>...</e1> and <e2>...</e2> tags.
2. It must contain atleast one valid word between the two entities.
   PS: Valid word excludes blankspace, stop words and punctutations.

Note: The major weakness of this method is that it cannot extract relations from sentences where there is no BETWEEN context.

### References and Citations:

Semi-Supervised Bootstrapping of Relationship Extractors with Distributional Semantics, EMNLP'15

@inproceedings{batista-etal-2015-semi,
    title = "Semi-Supervised Bootstrapping of Relationship Extractors with Distributional Semantics",
    author = "Batista, David S.  and Martins, Bruno  and Silva, M{\'a}rio J.",
    booktitle = "Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing",
    month = sep,
    year = "2015",
    address = "Lisbon, Portugal",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D15-1056",
    doi = "10.18653/v1/D15-1056",
    pages = "499--504",
}

"Large-Scale Semantic Relationship Extraction for Information Discovery" - Chapter 5, David S Batista, Ph.D. Thesis

@incollection{phd-dsbatista2016
  title = {Large-Scale Semantic Relationship Extraction for Information Discovery},
    author = {Batista, David S.},
  school = {Instituto Superior Técnico, Universidade de Lisboa},
  year = {2016}
}
