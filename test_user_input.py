import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter

from breds.test import BREDS as test_BREDS

def create_args() -> ArgumentParser:  # pylint: disable=missing-function-docstring
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sentence", 
        help="a sentence with at least two entities enclosed in <ENTITY> tags", 
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
        "--model",
        help="a trained model in the format of a .pkl file",
        type=str,
        required=False,
        default="model.pkl"
    )

    return parser

def main():

    parser = create_args()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    
    test_breads: test_BREDS
    test_breads = test_BREDS(args.word2vec)
    test_breads.load_model(args.model)
    test_breads.test_user_input(args.sentence)
        

if __name__ == "__main__":
    main()
