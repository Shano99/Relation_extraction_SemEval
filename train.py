import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter

from breds.bootstrapping import BREDS

def create_args() -> ArgumentParser:  # pylint: disable=missing-function-docstring
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--train_file", 
        help="a text file with a sentence per line, and with at least two entities per sentence", 
        type=str,
        required=True
    )
    parser.add_argument(
        "--word2vec",
        help="an embedding model based on word2vec, in the format of a .bin file",
        type=str,
        required=False,
        default="afp_apw_xin_embeddings.bin"
    )
    parser.add_argument(
        "--similarity",
        help="the minimum similarity between tuples and patterns to be considered a match",
        type=float,
        required=False,
        default=0.3
    )
    parser.add_argument(
        "--confidence",
        help="the minimum confidence score for a match to be considered a true positive",
        type=float,
        required=False,
        default=0.3
    )
    parser.add_argument(
        "--iterations",
        help="the minimum number of iterations to be executed",
        type=int,
        required=False,
        default=6,
    )
    parser.add_argument(
        "--alpha",
        help="weightage to BEFORE context",
        type=float,
        required=False,
        default=0.0,
    )
    parser.add_argument(
        "--beta",
        help="weightage to BETWEEN context",
        type=float,
        required=False,
        default=1.0,
    )
    parser.add_argument(
        "--gamma",
        help="weightage to AFTER context",
        type=float,
        required=False,
        default=0.0,
    )

    return parser

def main():

    parser = create_args()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    breads: BREDS

    breads = BREDS(
        args.word2vec,
        args.similarity,
        args.confidence,
        args.iterations,
        args.alpha,
        args.beta,
        args.gamma
    )
    
    breads.read_train_file(args.train_file)
    breads.generate_tuples()
    breads.init_bootstrap()
    breads.training_accuracy()

if __name__ == "__main__":
    main()
