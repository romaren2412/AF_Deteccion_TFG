import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bias", help="degree of non-IID to assign data to workers", type=float, default=0.1)
    parser.add_argument("--batch_size", help="batch size", default=-1, type=int)
    parser.add_argument("--lr", help="learning rate", default=-1, type=float)
    parser.add_argument("--nworkers", help="# workers", default=4, type=int)
    parser.add_argument("--nepochs", help="# epochs", default=-1, type=int)
    parser.add_argument("--gpu", help="index of gpu", default=0, type=int)
    parser.add_argument("--seed", help="seed", default=41, type=int)
    parser.add_argument("--nbyz", help="# byzantines", default=-1, type=int)
    parser.add_argument("--byz_type", help="type of attack", default='no', type=str,
                        choices=['no', 'mean_attack', 'label_flip', 'backdoor', 'dba'])

    # Engadidas por Roi
    parser.add_argument("--home_path", help="home path", default='', type=str)
    return parser.parse_args()


class Config:
    def __init__(self):
        args = parse_args()
        # GENERAL STUFF
        self.bias = 0.1
        self.gpu = 0
        self.seed = 41

        self.nbyz = args.nbyz
        self.byz_type = args.byz_type
        self.home_path = args.home_path

        self.SIZE = 10
        if args.byz_type == 'no':
            self.NBYZ = 0
        elif args.nbyz == -1:
            self.NBYZ = 3
        else:
            self.NBYZ = args.nbyz

        self.RANK = 0
        self.DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # DF FdAVG STUFF
        if args.batch_size == -1:
            self.BATCH_SIZE = 64
        else:
            self.BATCH_SIZE = args.batch_size

        self.FL_FREQ = 1

        if args.nepochs == -1:
            self.EPOCH = 100
        else:
            self.EPOCH = args.nepochs

        if args.lr == -1:
            self.LR = 1e-3
        else:
            self.LR = args.lr

        self.GLOBAL_LR = 1
