# -*- coding: utf-8 -*-
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import transforms


######################################################################
# DISTRIBUCIÓN DE DATOS
def repartir_datos(c, train_data, num_workers):
    # ASIGNACIÓN ALEATORIA DOS DATOS ENTRE OS CLIENTES
    # Semilla
    seed = c.seed
    np.random.seed(seed)

    bias_weight = c.bias
    other_group_size = (1 - bias_weight) / 9.
    worker_per_group = num_workers / 10

    each_worker_data = [[] for _ in range(num_workers)]
    each_worker_label = [[] for _ in range(num_workers)]

    # Iterar sobre el dataset completo
    for i in range(len(train_data)):
        data, label = train_data[i]
        data = data.view(1, 1, 28, 28)
        label = torch.tensor([label])

        # Asignar un punto de datos a un grupo
        upper_bound = label.item() * other_group_size + bias_weight
        lower_bound = label.item() * other_group_size
        rd = np.random.random_sample()

        if rd > upper_bound:
            worker_group = int(np.floor((rd - upper_bound) / other_group_size) + label.item() + 1)
        elif rd < lower_bound:
            worker_group = int(np.floor(rd / other_group_size))
        else:
            worker_group = label.item()

        # Asignar un punto de datos a un trabajador
        rd = np.random.random_sample()
        selected_worker = int(worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
        each_worker_data[selected_worker].append(data)
        each_worker_label[selected_worker].append(label)

    # Crear data loaders para cada trabajador
    client_data_loaders = []
    for data, labels in zip(each_worker_data, each_worker_label):
        dataset = torch.utils.data.TensorDataset(torch.cat(data, dim=0), torch.cat(labels, dim=0))
        data_loader = DataLoader(dataset, batch_size=c.BATCH_SIZE, shuffle=True,
                                 generator=torch.Generator(device='cuda'))
        client_data_loaders.append(data_loader)

    return client_data_loaders


def preparar_datos():
    # Definir la transformación para normalizar los datos
    transform = transforms.Compose([transforms.ToTensor()])
    # Cargar el conjunto de datos de entrenamiento
    train_data = torchvision.datasets.MNIST(root='MNIST/data', train=True, transform=transform, download=True)
    # Cargar el conjunto de datos de prueba
    test_data = torchvision.datasets.MNIST(root='MNIST/data', train=False, transform=transform, download=True)
    return train_data, test_data


def crear_root_dataset(c, train_data, num_clients, num_samples=100):
    target_size = 60000 // num_clients
    repeat_factor = target_size // num_samples
    root_subset = Subset(train_data, range(num_samples))
    repeated_subsets = [root_subset for _ in range(repeat_factor)]
    root_dataset = ConcatDataset(repeated_subsets)
    root_dataloader = DataLoader(root_dataset, batch_size=c.BATCH_SIZE, shuffle=True,
                                 generator=torch.Generator(device='cuda'))

    return root_dataloader
