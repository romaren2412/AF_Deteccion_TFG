import torch
import torchvision.transforms as transforms

import os
import numpy as np

from methods import SupervisedLearning
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch import nn


class TurtlebotTraining:
    def __init__(self, c) -> None:
        self.c = c
        transform = transforms.Compose(
            [transforms.ToTensor()])

        self.transform = transform

        self.LR = c.LR_tb
        self.BACH_SIZE = c.BACH_SIZE_tb

        self.device = c.DEVICE
        self.epochs = c.EPOCH_tb

        self.sl = SupervisedLearning(c, self)
        self.criterion = nn.MSELoss()

        self.selected_data = c.DATA_TB_IGUAIS if c.tipo_ben == c.tipo_mal else c.DATA_TB_5

        data_ubi, percent, partida = self.selected_data[self.c.RANK + 1]
        print(f'[INFO USER {self.c.RANK}] Cargando {data_ubi}')
        file_list = [os.path.join(data_ubi, file) for file in os.listdir(data_ubi) if
                     file.endswith('.txt') and not file.startswith('Read')]

        # Concatenamos los datos de todos los ficheros
        all_data = np.concatenate([np.loadtxt(file) for file in file_list])

        # Creamos la variable X que contiene las medidas del lidar y la variable y que contiene la velocidad lineal y angular
        # del robot para cada muestra de medidas

        init_range = int(percent * len(all_data))
        X = all_data[init_range * partida:init_range * (partida + 1) + 1, 2:]
        self.y = all_data[init_range * partida:init_range * (partida + 1) + 1, :2]

        for j, i in np.argwhere(X == np.inf):
            if X[j][i] == np.inf:
                if X[j][i - 1] < 0.50:
                    X[j][i] = 0.15
                elif X[j][i - 1] > 15:
                    X[j][i] = 30.0
                else:
                    X[j][i] = X[j][i - 1]

        X = 1 / X
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        self.X = (X - X_mean) / X_std

        self.X_train = self.y_train = self.X_test = self.y_test = self.x_test_global = self.y_test_global = None

    def create_train_test(self, index):
        print(f'[INFO USER {self.c.RANK}] Creando trainloader')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3,
                                                                                random_state=42)


        X_train = torch.from_numpy(self.X_train).type(torch.FloatTensor)
        y_train = torch.from_numpy(self.y_train).type(torch.FloatTensor)
        X_test = torch.from_numpy(self.X_test).type(torch.FloatTensor)
        y_test = torch.from_numpy(self.y_test).type(torch.FloatTensor)

        print(index, self.X_train.shape[0])
        self.df_train = TensorDataset(X_train, y_train, y_train)
        self.trainloader = DataLoader(self.df_train, batch_size=self.BACH_SIZE, shuffle=True,
                                                       generator=torch.Generator(device='cuda'))

        test_dataset = TensorDataset(X_test, y_test, y_test)
        self.testloader = DataLoader(test_dataset, batch_size=self.BACH_SIZE, shuffle=False,
                                                       generator=torch.Generator(device='cuda'))

    def init_global_test(self):
        return np.copy(self.X_test), np.copy(self.y_test)

    def continue_global_test(self, global_test_temp):
        x_test_global, y_test_global = global_test_temp
        x_test_global = np.concatenate((x_test_global, self.X_test))
        y_test_global = np.concatenate((y_test_global, self.y_test))
        return x_test_global, y_test_global

    def create_global_test(self, global_test_temp):
        x_test_global, y_test_global = global_test_temp
        x_test_global = torch.from_numpy(x_test_global).type(torch.FloatTensor)
        y_test_global = torch.from_numpy(y_test_global).type(torch.FloatTensor)
        test_dataset = TensorDataset(x_test_global, y_test_global, y_test_global)
        return DataLoader(test_dataset, batch_size=400, shuffle=False)