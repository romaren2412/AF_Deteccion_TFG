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
                        choices=['no', 'partial_trim', 'full_trim', 'mean_attack', 'full_mean_attack', 'gaussian',
                                 'dir_partial_krum_lambda', 'dir_full_krum_lambda', 'label_flip', 'backdoor', 'dba',
                                 'edge', 'backdoor_sen_pixel'])
    parser.add_argument("--aggregation", help="aggregation rule", default='fedavg', type=str,
                        choices=['fedavg', 'flare'])

    # Engadidas por Roi
    parser.add_argument("--home_path", help="home path", default='', type=str)

    parser.add_argument("--tipo_exec", help="tipo de execucion", default='detect', type=str,
                        choices=['detect', 'loop', 'no_detect'])
    parser.add_argument("--silhouette", help="medida de confianza necesaria (silhouette)", default=0.0,
                        type=float)
    parser.add_argument("--det_start", help="Inicio de deteccion", default=-1, type=int)
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
        self.aggregation = args.aggregation
        self.home_path = args.home_path
        self.tipo_exec = args.tipo_exec
        self.silhouette = args.silhouette

        if args.det_start == -1:
            self.det_start = 5
        else:
            self.det_start = args.det_start

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

        self.FL_FREQ = 5

        if args.nepochs == -1:
            self.EPOCH = 20
        else:
            self.EPOCH = args.nepochs

        if args.lr == -1:
            self.LR = 4e-2
        else:
            self.LR = args.lr
