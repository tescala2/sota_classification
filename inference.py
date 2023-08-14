from argparse import ArgumentParser
from utils.loops import run_inference


def process_args():
    parser = ArgumentParser(description='Train and Evaluate Model on Dataset')
    parser.add_argument("-dataset",
                        help="Input to determine dataset, can be a directory or a name for lookup in config file",
                        default=None)
    parser.add_argument("-name", dest="model",
                        help="Input to determine model, can be a file, directory or a name for lookup in config file",
                        default=None)
    parser.add_argument("-hardware", dest="hardware",
                        default="auto",
                        help="Select Hardware option to use, most likely gpus (default: local machine least used gpu)")
    parser.add_argument("-bs",
                        default=None,
                        help="Override batch size in -trn_cfg_file.")
    return parser.parse_args()


if __name__ == '__main__':
    args = process_args()

    num_classes = 10

    run_inference(data_path=args.dataset,
                  model_name=args.name,
                  device=args.hardware,
                  bs=args.bs,
                  num_classes=num_classes)
