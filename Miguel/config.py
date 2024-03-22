import numpy as np
import torch


class config:
    def __init__(self):
        # GENERAL STUFF
        self.SIZE = 4
        self.RANK = 0

        self.CONTINUE_TRAINING = 1
        self.SAVE_NAME_RESULTS = '../probas/_test9/'
        self.TIMES = 1
        self.PROBA_PESOS_EXPECIFICOS = False
        self.UBI_PROBA_PESOS_EXPECIFICOS = '../probas/no_conv_pesos/weight1.txt'
        self.DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.TRAINING = "DF_FdAVG"
        self.rober_flag = False

        # COMUNICATION STUFF
        self.SERVER_SOCKET = "wlo1"  # "eno2" #wlo1
        self.CLIENT_SOCKET = "wlo1"  # "wlan0"

        # DF FdAVG STUFF
        self.DATA_UBI = "../data/d5/"
        self.BACH_SIZE = 50
        self.EPOCH = 500
        self.SYNC_EP = 1
        self.LR = 1e-4
        self.recalculate_db()
        # config by hand

        self.CONTINOUS = True
        self.CHANGING_DATA = False
        self.EPOCH_CHANGE = 24

        self.PREMODEL_EPOCH = 10
        self.POSTMODEL_EPOCH = self.EPOCH - self.PREMODEL_EPOCH
        self.NUMBER_CLASES = 10

        self.GLOCAL = False

        self.bayesian_dropout = False

        self.path_tomodels_zero = '../Miguel/models/d5_net_zeros.pt'

        self.FLDET_START = 50

    def recalculate_db(self):

        ranks = {1: np.arange(60, 69).tolist(),
                 2: np.arange(69, 78).tolist(),
                 3: np.arange(78, 88).tolist(),
                 4: np.arange(88, 97).tolist(),
                 5: np.arange(97, 107).tolist(), }

        ranks_test = {1: np.arange(107, 119).tolist(),
                      2: np.arange(107, 119).tolist(),
                      3: np.arange(107, 119).tolist(),
                      4: np.arange(107, 119).tolist(),
                      5: np.arange(107, 119).tolist(), }

        self.TRAIN_INDEX = np.array(ranks[self.RANK + 1])
        self.TEST_INDEX_LOCAL = np.array(ranks_test[self.RANK + 1])
        self.TEST_INDEX_GLOBAL = []
        for v in ranks_test.values():
            self.TEST_INDEX_GLOBAL += v
            break

        """
        # REDUCIR O NÚMERO DE DATOS, IDÉNTICAMENTE DISTRIBUIDOS
        self.TRAIN_INDEX = self.TRAIN_INDEX[:int(len(self.TRAIN_INDEX) / 2)]
        self.TEST_INDEX_LOCAL = self.TEST_INDEX_LOCAL[:int(len(self.TEST_INDEX_LOCAL) / 2)]
        self.TEST_INDEX_GLOBAL = self.TEST_INDEX_GLOBAL[:int(len(self.TEST_INDEX_GLOBAL) / 2)]
        """
