import argparse
import tqdm


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest="command")

    # Arguments for 'prepare_data' command
    parser_prepare_data = subparsers.add_parser("prepare_data", help="Prepare dataset for training and evaluation.")
    parser_prepare_data.add_argument("--feats_dataset_id", type=str, default="01J6KF3JRCATRJQ9CPJTRV5VBM", help="Beaker ID containing the extracted lexical features and subsets.")
    parser_prepare_data.add_argument("--reference_dataset_id", type=str, default="01J6KBM2VCM9EQ7MER26VBXCCM", help="Beaker ID containing the metadata features.")
    parser_prepare_data.add_argument("--dataset")
    # fmt: on
    pass


def main():
    pass


if __name__ == "__main__":
    main()
