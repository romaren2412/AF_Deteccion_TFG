# -*- coding: utf-8 -*-
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


######################################################################
# DISTRIBUCIÓN DE DATOS
def repartir_datos(args, train_data_loader, num_workers, device):
    # ASIGNACIÓN ALEATORIA DOS DATOS ENTRE OS CLIENTES
    # Semilla
    seed = args.seed
    np.random.seed(seed)

    bias_weight = args.bias
    other_group_size = (1 - bias_weight) / 9.
    worker_per_group = num_workers / 10

    each_worker_data = [[] for _ in range(num_workers)]
    each_worker_label = [[] for _ in range(num_workers)]

    for _, (data, label) in enumerate(train_data_loader):
        data, label = data.to(device), label.to(device)
        for (x, y) in zip(data, label):
            x = x.to(device).view(1, 1, 28, 28)
            y = y.to(device).view(-1)

            # Asignar un punto de datos a un grupo
            upper_bound = (y.cpu().numpy()) * other_group_size + bias_weight
            lower_bound = (y.cpu().numpy()) * other_group_size
            rd = np.random.random_sample()

            if rd > upper_bound:
                worker_group = int(np.floor((rd - upper_bound) / other_group_size) + y.cpu().numpy() + 1)
            elif rd < lower_bound:
                worker_group = int(np.floor(rd / other_group_size))
            else:
                worker_group = y.cpu().numpy()

            # Asignar un punto de datos a un trabajador
            rd = np.random.random_sample()
            selected_worker = int(worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
            each_worker_data[selected_worker].append(x)
            each_worker_label[selected_worker].append(y)

    # Concatenar los datos para cada trabajador para evitar huecos
    each_worker_data = [torch.cat(each_worker, dim=0) for each_worker in each_worker_data]
    each_worker_label = [torch.cat(each_worker, dim=0) for each_worker in each_worker_label]

    # Barajar aleatoriamente los trabajadores
    random_order = np.random.RandomState(seed=seed).permutation(num_workers)
    each_worker_data = [each_worker_data[i] for i in random_order]
    each_worker_label = [each_worker_label[i] for i in random_order]

    return each_worker_data, each_worker_label


def preparar_datos():
    # Definir la transformación para normalizar los datos
    transform = transforms.Compose([transforms.ToTensor()])
    # Cargar el conjunto de datos de entrenamiento
    train_data = torchvision.datasets.MNIST(root='00_MNIST/data_MNIST', train=True, transform=transform, download=True)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=60000, shuffle=True,
                                                    generator=torch.Generator(device='cuda'))
    # Cargar el conjunto de datos de prueba
    test_data = torchvision.datasets.MNIST(root='00_MNIST/data_MNIST', train=False, transform=transform, download=True)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=500, shuffle=False)
    return train_data_loader, test_data_loader, test_data
