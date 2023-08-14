from argparse import ArgumentParser
from utils.loops import run


def process_args():
    parser = ArgumentParser(description='Train and Evaluate Model on Dataset')
    parser.add_argument("-dataset",
                        dest='dataset',
                        help="Input to determine dataset, can be a directory or a name for lookup in config file",
                        default=None)
    parser.add_argument("-name",
                        dest="name",
                        help="Input to determine model, can be a file, directory or a name for lookup in config file",
                        default=None)
    parser.add_argument("-hardware",
                        dest="hardware",
                        default="auto",
                        help="Select Hardware option to use, most likely gpus (default: local machine least used gpu)")
    parser.add_argument("-ep",
                        dest="epochs",
                        default=15,
                        help="number to epochs to resume training for")
    parser.add_argument("-bs",
                        default=64,
                        help="Override batch size in -trn_cfg_file.")
    parser.add_argument("-cont",
                        default=False,
                        help="Continue training from trained weights")
    return parser.parse_args()


if __name__ == '__main__':
    args = process_args()

    num_classes = 10

    run(data_path=args.dataset,
        model_name=args.name,
        device=args.hardware,
        epochs=args.epochs,
        bs=args.bs,
        cont=args.cont,
        num_classes=num_classes,
        lr=0.001,
        progress=True,
        parallel=False)
