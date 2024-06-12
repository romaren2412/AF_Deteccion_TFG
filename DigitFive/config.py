import argparse

import numpy as np
import torch

diccionario = {'mnist': np.arange(0, 59).tolist(),
               'mnistm': np.arange(60, 118).tolist(),
               'svhn': np.arange(119, 191).tolist(),
               'syn': np.arange(192, 203).tolist(),
               'usps': np.arange(204, 212).tolist()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="index of gpu", default=0, type=int)
    parser.add_argument("--nbyz", help="number of byzantine workers", default=1, type=int)
    parser.add_argument("--lr", help="learning rate", default=-1, type=float)

    # Engadidas por Roi
    parser.add_argument("--home_path", help="home path", default='', type=str)
    parser.add_argument("--data_type", help="tipo de datos", default='mnistm', type=str,
                        choices=['mnist', 'svhn', 'syn', 'usps', 'mnistm'])
    parser.add_argument("--extra_data_type", help="tipo de datos extra", default='mnist', type=str,
                        choices=['mnist', 'svhn', 'syn', 'usps', 'mnistm'])

    return parser.parse_args()


class Config:
    def __init__(self, opt='1b'):
        # OPT: '1b' para 1Batch, 'mini' para Minibatches
        args = parse_args()
        # GENERAL STUFF
        self.SIZE = 5
        self.RANK = 0
        self.gpu = args.gpu

        self.NBYZ = args.nbyz
        self.home_path = args.home_path

        self.data_type = args.data_type
        self.extra_data_type = args.extra_data_type

        self.DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # DF FdAVG STUFF
        self.DATA_UBI = "./data/d5/"
        self.BATCH_TEST_SIZE = 5000
        if opt == '1b':
            self.BATCH_SIZE = 11000
            self.EPOCH = 300

            if args.lr == -1:
                self.LR = 1
            else:
                self.LR = args.lr
        elif opt == 'mini':
            self.BATCH_SIZE = 32
            self.EPOCH = 20

            if args.lr == -1:
                self.LR = 0.1
            else:
                self.LR = args.lr
        else:
            raise ValueError("Opción no válida para el tipo de ejecución.")

        self.GLOBAL_LR = 1

    def create_ranks(self, num_clients=4):
        key = self.data_type
        extra_key = self.extra_data_type
        """
        Divide los índices de un tipo de datos principal en un 80-20% para formar conjuntos de entrenamiento y prueba.
        Luego distribuye esos índices entre los clientes regulares. Posteriormente, asigna índices del extra_key al
        cliente extra, basado en el tamaño promedio de las particiones de los clientes regulares.
    
        Args:
        diccionario (dict): Diccionario con listas de índices para cada tipo de dato.
        key (str): Clave principal cuyos índices se van a dividir y distribuir entre los clientes regulares.
        extra_key (str): Clave extra cuyos índices serán para el cliente extra.
        num_clients (int): Número total de clientes regulares.
    
        Returns:
        dict: Diccionarios con los rangos de entrenamiento y prueba para cada cliente, incluido el cliente extra.
        """
        indices = diccionario[key]
        np.random.shuffle(indices)  # Desordenar para asegurar aleatoriedad en la selección.

        # Calcular el punto de corte para el 80-20%
        split_index = int(len(indices) * 0.8) if len(indices) > 5 else len(indices)  # Asegura un mínimo para la prueba

        # Dividir los índices en conjuntos de entrenamiento y prueba
        train_indices = indices[:split_index]
        test_indices = indices[split_index:]

        # Asegurar al menos un índice por cliente en entrenamiento y prueba, repitiendo índices si es necesario
        while len(train_indices) < num_clients:
            train_indices = np.concatenate([train_indices, indices[:num_clients - len(train_indices)]])

        while len(test_indices) < num_clients:
            test_indices = np.concatenate(
                [test_indices, indices[split_index:split_index + num_clients - len(test_indices)]])

        # Calcular el tamaño de las particiones para entrenamiento y prueba
        partition_size_train = len(train_indices) // num_clients
        partition_size_test = len(test_indices) // num_clients

        # Distribuir los índices de entrenamiento y prueba entre los clientes regulares
        ranks_train = {}
        ranks_test = {}

        start_train = 0
        start_test = 0
        for i in range(1, num_clients + 1):
            end_train = start_train + partition_size_train + (1 if i <= len(train_indices) % num_clients else 0)
            end_test = start_test + partition_size_test + (1 if i <= len(test_indices) % num_clients else 0)

            ranks_train[i] = train_indices[start_train:end_train]
            ranks_test[i] = test_indices[start_test:end_test]

            start_train = end_train
            start_test = end_test

        # Extraer índices para el cliente extra basado en el tamaño promedio de las particiones
        extra_indices = diccionario[extra_key]
        np.random.shuffle(extra_indices)  # Desordenar para aleatoriedad
        split_extra_index = int(len(extra_indices) * 0.8)
        extra_train_indices = extra_indices[:split_extra_index]
        extra_test_indices = extra_indices[split_extra_index:]

        # Asegurar al menos un índice en entrenamiento y prueba para el cliente extra
        if len(extra_train_indices) < 1:
            extra_train_indices = np.concatenate([extra_train_indices, extra_indices[:1]])
        if len(extra_test_indices) < 1:
            extra_test_indices = np.concatenate(
                [extra_test_indices, extra_indices[split_extra_index:split_extra_index + 1]])

        # Asignar índices de entrenamiento y prueba al cliente extra
        ranks_train[0] = extra_train_indices[:partition_size_train]
        ranks_test[0] = extra_test_indices[:partition_size_test]

        # Recortar los diccionarios a la longitud mínima para igualar el tamaño de las particiones
        min_length_train = min((len(lista) for lista in ranks_train.values()))
        cut_ranks_train = {k: v[:min_length_train] for k, v in ranks_train.items()}
        min_length_test = min((len(lista) for lista in ranks_test.values()))
        cut_ranks_test = {k: v[:min_length_test] for k, v in ranks_test.items()}
        return cut_ranks_train, cut_ranks_test

    def seleccionar_indices_root(self, num_data=4, num_extra=1):
        data_type = self.data_type
        extra_data_type = self.extra_data_type

        # Obtener todos los índices disponibles para data y extra_data
        all_data_indices = diccionario[data_type]
        all_extra_data_indices = diccionario[extra_data_type]

        # Mezclar los índices para asegurar aleatoriedad
        np.random.shuffle(all_data_indices)
        np.random.shuffle(all_extra_data_indices)

        # Seleccionar índices de entrenamiento y prueba sin superposición
        train_data_indices = all_data_indices[:num_data]
        test_data_indices = all_data_indices[num_data:num_data * 2]
        train_extra_data_indices = all_extra_data_indices[:num_extra]
        test_extra_data_indices = all_extra_data_indices[num_extra:num_extra * 2]

        # Combinar los índices de entrenamiento y prueba
        combined_train_indices = list(train_extra_data_indices) + list(train_data_indices)
        combined_test_indices = list(test_extra_data_indices) + list(test_data_indices)

        return combined_train_indices, combined_test_indices
