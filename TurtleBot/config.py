import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="index of gpu", default=0, type=int)
    parser.add_argument("--byz_type", help="type of attack", default='no', type=str,
                        choices=['no', 'partial_trim', 'full_trim', 'mean_attack', 'full_mean_attack', 'gaussian',
                                 'dir_partial_krum_lambda', 'dir_full_krum_lambda', 'label_flip', 'backdoor', 'dba',
                                 'edge', 'backdoor_sin_pixel'])
    parser.add_argument("--aggregation", help="aggregation rule", default='median', type=str,
                        choices=['simple_mean', 'trim', 'krum', 'median'])
    parser.add_argument("--nbyz", help="number of byzantine workers", default=0, type=int)

    # Engadidas por Roi
    parser.add_argument("--home_path", help="home path", default='', type=str)
    parser.add_argument("--tipo_exec", help="tipo de execución", default='detect', type=str,
                        choices=['detect', 'loop', 'no_detect'])
    # learning rate
    parser.add_argument("--lr", help="learning rate", default=-1, type=float)
    # tipo de datos
    parser.add_argument("--tipo_ben", help="tipo de datos benignos", default='1', type=str,
                        choices=['1', '2', 'der', 'izq'])
    parser.add_argument("--tipo_mal", help="tipo de datos maliciosos", default='2', type=str,
                        choices=['1', '2', 'der', 'izq'])
    return parser.parse_args()


class Config:
    def __init__(self):
        args = parse_args()
        # GENERAL STUFF
        self.SIZE = 5
        self.RANK = 0

        self.DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # self.BACH_SIZE_tb = 256
        self.BACH_SIZE_tb = 560
        # self.EPOCH_tb = 201
        self.EPOCH_tb = 2000
        if args.lr == -1:
            self.LR_tb = 0.5
        else:
            self.LR_tb = args.lr

        self.FLDET_START = 50

        self.DATA_TB = {
            1: ("../data/datos_turtlebot_1/", 0.1, 0),
            2: ("../data/datos_turtlebot_1/", 0.1, 1),
            3: ("../data/datos_turtlebot_1/", 0.1, 2),
            4: ("../data/datos_turtlebot_1/", 0.1, 3),
            5: ("../data/datos_turtlebot_2/", 0.1, 0),
            6: ("../data/datos_turtlebot_der/", 0.1, 0),
            7: ("../data/datos_turtlebot_izq/", 0.1, 0)}

        data_dic = {
            '1': "../data/datos_turtlebot_1/",
            '2': "../data/datos_turtlebot_2/",
            'der': "../data/datos_turtlebot_der/",
            'izq': "../data/datos_turtlebot_izq/"
        }

        data_ben = data_dic[args.tipo_ben]
        data_mal = data_dic[args.tipo_mal]

        self.DATA_TB_5 = {
            1: (data_mal, 0.1, 0),
            2: (data_ben, 0.1, 0),
            3: (data_ben, 0.1, 1),
            4: (data_ben, 0.1, 2),
            5: (data_ben, 0.1, 3)}

        self.DATA_TB_4_Dif = {
            1: (data_dic['1'], 0.1, 0),
            2: (data_dic['2'], 0.1, 0),
            3: (data_dic['der'], 0.1, 0),
            4: (data_dic['izq'], 0.1, 0)}
