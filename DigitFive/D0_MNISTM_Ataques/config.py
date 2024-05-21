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
    return parser.parse_args()


class Config:
    def __init__(self):
        args = parse_args()
        # GENERAL STUFF
        self.SIZE = 10
        self.RANK = 0

        self.DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # DF FdAVG STUFF
        self.BATCH_TEST_SIZE = 1000
        self.BATCH_SIZE = 5000
        self.EPOCH = 1000
        if args.lr == -1:
            self.LR = 1
        else:
            self.LR = args.lr

        self.recalculate_db()

        self.FLDET_START = 50

    def recalculate_db(self):
        # MNISTM: 60 - 117
        total_range = np.arange(60, 118)  # 118 porque arange no incluye el último valor

        # Tamaño de cada partición
        partition_size = len(total_range) // self.SIZE
        remainder = len(total_range) % self.SIZE  # Resto para distribuir los índices extras

        # Dividir los índices en self.SIZE partes
        partitions = []
        start_idx = 0
        for i in range(self.SIZE):
            end_idx = start_idx + partition_size + (
                1 if i < remainder else 0)  # Añadir un índice extra a las primeras 'remainder' particiones
            partitions.append(total_range[start_idx:end_idx])
            start_idx = end_idx

        # Asignar índices de entrenamiento y prueba basados en el rango correspondiente
        self.TRAIN_INDEX = np.array(partitions[self.RANK])
        self.TEST_INDEX_LOCAL = np.array(
            partitions[self.RANK])  # Esto podría ajustarse si necesitas diferentes índices para prueba

        # Crear índices de prueba globales combinando todos los índices de prueba locales
        self.TEST_INDEX_GLOBAL = np.concatenate(partitions)  # O ajustar según se necesite

        # Opcional: Reducir el número de datos si es necesario
        # self.TRAIN_INDEX = self.TRAIN_INDEX[:3]
        # self.TEST_INDEX_LOCAL = self.TEST_INDEX_LOCAL[:3]
        # self.TEST_INDEX_GLOBAL = self.TEST_INDEX_GLOBAL[:3]
