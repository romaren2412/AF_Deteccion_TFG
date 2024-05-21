import numpy as np
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="index of gpu", default=0, type=int)
    parser.add_argument("--byz_type", help="type of attack", default='no', type=str,
                        choices=['no', 'partial_trim', 'full_trim', 'mean_attack', 'full_mean_attack', 'gaussian',
                                 'dir_partial_krum_lambda', 'dir_full_krum_lambda', 'label_flip', 'backdoor', 'dba',
                                 'edge', 'backdoor_sen_pixel'])
    parser.add_argument("--aggregation", help="aggregation rule", default='simple_mean', type=str,
                        choices=['simple_mean', 'trim', 'krum', 'median'])
    parser.add_argument("--nbyz", help="number of byzantine workers", default=1, type=int)
    parser.add_argument("--lr", help="learning rate", default=-1, type=float)

    # Engadidas por Roi
    parser.add_argument("--home_path", help="home path", default='', type=str)
    parser.add_argument("--tipo_exec", help="tipo de execución", default='detect', type=str,
                        choices=['detect', 'loop', 'no_detect'])
    parser.add_argument("--silhouette", help="medida de confianza necesaria (silhouette)", default=0.0,
                        type=float)
    parser.add_argument("--net", help="tipo de rede", default='Miguel', type=str,
                        choices=['CNN', 'Miguel'])
    parser.add_argument("--data_type", help="tipo de datos", default='mnistm', type=str,
                        choices=['mnist', 'svhn', 'syn', 'usps', 'mnistm'])
    parser.add_argument("--extra_data_type", help="tipo de datos extra", default='mnist', type=str,
                        choices=['mnist', 'svhn', 'syn', 'usps', 'mnistm'])

    return parser.parse_args()


class Config:
    def __init__(self):
        args = parse_args()
        # GENERAL STUFF
        self.SIZE = 5
        self.RANK = 0

        self.DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # DF FdAVG STUFF
        self.BATCH_TEST_SIZE = 1000
        self.BATCH_SIZE = 5000
        self.EPOCH = 1000
        if args.lr == -1:
            self.LR = 1
            self.LR_CNN = 0.2
        else:
            self.LR = args.lr
            self.LR_CNN = args.lr

        self.FLDET_START = 50

    def recalculate_db_5bases(self):
        # PROBA 1
        ranks = {1: np.arange(0, 49).tolist(),  # mnist m
                 2: np.arange(60, 107).tolist(),  # mnist m
                 3: np.arange(120, 179).tolist(),  # svhn
                 4: np.arange(192, 200).tolist(),  # syn
                 5: np.arange(204, 209).tolist()}  # usps

        ranks_test = {1: np.arange(50, 59).tolist(),
                      2: np.arange(107, 118).tolist(),
                      3: np.arange(180, 191).tolist(),
                      4: np.arange(201, 203).tolist(),
                      5: np.arange(210, 212).tolist(), }

        self.TRAIN_INDEX = np.array(ranks[self.RANK + 1])
        self.TEST_INDEX_LOCAL = np.array(ranks_test[self.RANK + 1])
        self.TEST_INDEX_GLOBAL = []
        for v in ranks_test.values():
            self.TEST_INDEX_GLOBAL.append(np.random.choice(v))

        self.TRAIN_INDEX = self.TRAIN_INDEX[:3]
        self.TEST_INDEX_LOCAL = self.TEST_INDEX_LOCAL[:3]
