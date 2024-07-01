# BRED - BOOTSTRAPPING - NLP

## Description

BREDS extracts relationships using a bootstrapping/semi-supervised approach, it relies on an initial set of seeds, i.e. pairs of named-entities representing relationship type to be extracted.  

The algorithm expands the initial set of seeds using distributional semantics to generalize the relationship while 
limiting the semantic drift.


## Sample



## Code changes and enhancements

## Fine tuning model

## Evaluation

## Default configuration and arguments


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
