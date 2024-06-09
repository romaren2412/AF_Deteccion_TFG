import numpy as np
from config import Config

c = Config()

#######################################################################################################################
# TARGETED #


def backdoor(data, target_backdoor_dba):
    # Igual que backdoor_range en MLR pero sin redimensionar a (-1,784)
    # 2. BACKDOOR: introducir un patrón específico nas mostras de datos para engañar ao modelo central.
    """
    :param data: conxunto de input, label
    :param target_backdoor_dba: ETIQUETA DO PATRÓN
    :return: each_worker_data e each_worker_label actualizados
    """
    # Duplica a primeira metade de mostras de datos de cada traballador byzantino
    inputs, labels = data
    # REDIMENSIONAR OS DATOS PARA O BUCLE
    met = len(inputs) // 2
    prim = inputs[:met]  # {tensor(X, 3, 16, 16)}
    inputs = prim.repeat(2, 1, 1, 1)  # {tensor(2X, 3, 16, 16)}
    labels = labels[:met].repeat(2)
    inputs = np.transpose(inputs, (0, 2, 3, 1))

    # Modificar patrones específicos en las muestras de datos duplicadas (esquina inferior derecha)
    for example_id in range(0, inputs.shape[0], 2):
        inputs[example_id][26][26] = 1
        inputs[example_id][24][26] = 1
        inputs[example_id][26][24] = 1
        inputs[example_id][25][25] = 1
        # Etiqueta 0 para el patrón
        labels[example_id] = target_backdoor_dba

    inputs = np.transpose(inputs, (0, 3, 1, 2))
    return inputs, labels


def dba(data, target, index):
    # DUPLICAR OS DATOS ORIXINAIS
    inputs, labels = data
    met = len(inputs) // 2
    inputs = inputs[:met].repeat(2, 1, 1, 1)
    inputs = np.transpose(inputs, (0, 2, 3, 1))
    labels = labels[:met].repeat(2)
    pixeles = {0: (26, 26), 1: (24, 16), 2: (26, 24), 3: (25, 25)}
    if index in pixeles:
        for example_id in range(0, inputs.shape[0], 2):
            inputs[example_id][pixeles[index][0]][pixeles[index][1]] = 1
            labels[example_id] = target

    inputs = np.transpose(inputs, (0, 3, 1, 2))
    return inputs, labels


def label_flip(data):
    inputs, labels = data
    labels = (labels + 1) % 9
    return inputs, labels

#######################################################################################################################
# UNTARGETED #


def mean_attack(model):
    for param in model.parameters():
        param.data = -param.data


def scaling_attack(model, scaling_factor=c.SIZE):
    for param in model.parameters():
        param.data = param.data * scaling_factor
    return model
