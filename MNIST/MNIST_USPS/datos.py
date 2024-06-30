# -*- coding: utf-8 -*-
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.data import Subset

root_data = './data'


######################################################################
# DISTRIBUCIÓN DE DATOS
def repartir_datos(args, loader_mnist, loader_usps, num_workers, device, undetected_byz_index):
    seed = args.seed
    np.random.seed(seed)
    bias_weight = args.bias
    other_group_size = (1 - bias_weight) / 9.

    # Listas para almacenar datos y etiquetas para cada trabajador
    each_worker_data = [[] for _ in range(num_workers)]
    each_worker_label = [[] for _ in range(num_workers)]

    # Función para distribuir datos a un cargador de datos específico considerando bias_weight
    def distribuir_data(loader, indices_objetivo):
        num_workers = len(indices_objetivo)
        worker_per_group = num_workers // 10
        for _, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            for i, (x, y) in enumerate(zip(data, label)):
                x = x.to(device).view(1, 1, 28, 28)
                y = y.to(device).view(-1)

                # Asignar un punto de datos a un grupo en función de bias_weight
                upper_bound = (y.cpu().item()) * other_group_size + bias_weight
                lower_bound = (y.cpu().item()) * other_group_size
                rd = np.random.random_sample()

                if rd > upper_bound:
                    worker_group = int(np.floor((rd - upper_bound) / other_group_size) + y.cpu().item() + 1)
                elif rd < lower_bound:
                    worker_group = int(np.floor(rd / other_group_size))
                else:
                    worker_group = y.cpu().item()

                # Asegurar que el grupo seleccionado pertenece al conjunto de índices objetivo
                worker_group_indices = [idx for idx in indices_objetivo if idx % worker_per_group == worker_group]
                if worker_group_indices:
                    selected_worker = np.random.choice(worker_group_indices)
                else:
                    selected_worker = np.random.choice(indices_objetivo)
                each_worker_data[selected_worker].append(x)
                each_worker_label[selected_worker].append(y)

    # Identificar índices de clientes benignos
    indices_benignos = [i for i in range(num_workers) if i not in undetected_byz_index]

    # Distribuir datos MNIST a clientes benignos y USPS a clientes bizantinos
    distribuir_data(loader_mnist, indices_benignos)
    distribuir_data(loader_usps, undetected_byz_index)

    # Concatenar los datos para cada trabajador para evitar huecos
    each_worker_data = [torch.cat(each_worker, dim=0) for each_worker in each_worker_data]
    each_worker_label = [torch.cat(each_worker, dim=0) for each_worker in each_worker_label]

    return each_worker_data, each_worker_label


def preparar_datos_mnist_usps(num_workers, args):
    tamanho_orixinal = 60000
    tamanho_mnist = int(tamanho_orixinal * (num_workers - args.nbyz) / num_workers)
    tamanho_usps = int(tamanho_orixinal * args.nbyz / num_workers)
    return preparar_mnist(tamanho_mnist), preparar_usps(tamanho_usps)

def preparar_mnist(tamanho_mnist):
    # Definir la transformación para normalizar los datos
    transform = transforms.Compose([transforms.ToTensor()])
    train_data_mnist = torchvision.datasets.MNIST(root=root_data, train=True, transform=transform, download=True)
    indices_mnist = torch.randperm(len(train_data_mnist))[:tamanho_mnist]
    subset_mnist = Subset(train_data_mnist, indices_mnist)
    train_data_loader_mnist = torch.utils.data.DataLoader(subset_mnist, shuffle=True,
                                                          generator=torch.Generator(device='cuda'))
    return train_data_loader_mnist

def preparar_usps(tamanho_usps):
    # Transformación para USPS, incluyendo el cambio de tamaño a 28x28
    transform_usps = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    # Carga del conjunto de datos USPS
    train_data_usps = datasets.USPS(root=root_data, train=True, transform=transform_usps, download=True)

    # Verifica si necesitas duplicar los datos para alcanzar tamanho_usps
    if len(train_data_usps) < tamanho_usps:
        num_replicas = tamanho_usps // len(train_data_usps) + 1
        indices_usps = np.tile(np.arange(len(train_data_usps)), num_replicas)[:tamanho_usps]
    else:
        indices_usps = np.random.choice(np.arange(len(train_data_usps)), tamanho_usps, replace=False)
    subset_usps = Subset(train_data_usps, indices_usps)

    # DataLoader para el subconjunto de USPS
    train_data_loader_usps = DataLoader(subset_usps, shuffle=True,
                                        generator=torch.Generator(device='cuda'))
    return train_data_loader_usps


def preparar_test():
    transform = transforms.Compose([transforms.ToTensor()])
    transform_usps = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    test_data_mnist = torchvision.datasets.MNIST(root=root_data, train=False, transform=transform, download=True)
    test_data_usps = torchvision.datasets.USPS(root=root_data, train=False, transform=transform_usps, download=True)
    combined_test_data = torch.utils.data.ConcatDataset([test_data_mnist, test_data_usps])
    combined_test_loader = torch.utils.data.DataLoader(combined_test_data, batch_size=500, shuffle=False)
    return combined_test_loader
