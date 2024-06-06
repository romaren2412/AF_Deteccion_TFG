import numpy as np
import torch
import math


def select_byzantine_range(t):
    if t == 'partial_trim':
        return trim_partial
    elif t == 'full_trim':
        return trim_full
    elif t == 'dir_partial_krum_lambda':
        return krum_partial
    elif t == 'dir_full_krum_lambda':
        return krum_full
    elif t == 'mean_attack':
        return mean_attack
    elif t == 'backdoor':
        return scaling_attack
    elif t == 'no':
        return no_byz
    else:
        raise NotImplementedError


def no_byz(v, undetected_byz):
    return v


def mean_attack(v, undetected_byz):
    for i in undetected_byz:
        v[i] = -v[i]
    return v


def backdoor(each_worker_data, each_worker_label, undetected_byz, target_backdoor_dba):
    # 1. BACKDOOR: introducir un patrón específico nas mostras de datos para engañar ao modelo central.
    """
    :param each_worker_data:
    :param each_worker_label:
    :param undetected_byz: lista de clientes byzantinos
    :param target_backdoor_dba: ETIQUETA DO PATRÓN
    :return: each_worker_data e each_worker_label actualizados
    """
    for i in undetected_byz:
        met = len(each_worker_data[i]) // 2
        # Duplica as primeiras 300 mostras de datos de cada traballador byzantino
        # REDIMENSIONAR OS DATOS PARA O BUCLE
        prim = each_worker_data[i][:met]  # {tensor(X, 784)}
        sec = prim.view(-1, 1, 28, 28)  # {tensor(X, 1, 28, 28)}
        each_worker_data[i] = sec.repeat(2, 1, 1, 1)  # {tensor(2X, 1, 28, 28)}
        each_worker_label[i] = each_worker_label[i][:met].repeat(2)

        # Modifica patróns específicos nas mostras de datos duplicadas (esquina inferior dereita)
        # Segundo este bucle:
        #   each_worker_data[i] tería que ser tensor(600, 1, 28, 28)
        #   each_worker_label[i] tería que ser tensor(600)
        for example_id in range(0, each_worker_data[i].shape[0], 2):
            each_worker_data[i][example_id][0][26][26] = 1
            each_worker_data[i][example_id][0][24][26] = 1
            each_worker_data[i][example_id][0][26][24] = 1
            each_worker_data[i][example_id][0][25][25] = 1
            # Etiqueta 0 para o patrón
            each_worker_label[i][example_id] = target_backdoor_dba

    return each_worker_data, each_worker_label


def scaling_attack(v, undetected_byz):
    factor = len(v)
    for i in undetected_byz:
        v[i] = factor * v[i]
    return v


###########################################################################################
# ATAQUES A MÉTODOS ROBUSTOS
###########################################################################################
def trim_partial(v, undetected_byz):
    """
    Partial-knowledge Trim attack.
    :param v: lista de gradientes
    :param undetected_byz: lista de dispositivos byzantinos
    """
    f = len(undetected_byz)
    vi_shape = v[0].shape
    todos_grads = torch.cat(v, dim=1)
    byz_grads = todos_grads[:, :f]
    e_mu = torch.mean(byz_grads, dim=1)
    e_sigma = torch.sqrt(torch.sum((byz_grads - e_mu.view(-1, 1)) ** 2, dim=1) / f)

    # APLICAR TÉCNICA NOS DISPOSITIVOS COMPROMETIDOS
    for i in undetected_byz:
        norm = torch.norm(v[i])
        v[i] = (e_mu - e_sigma * torch.sign(e_mu)).view(vi_shape)
        v[i] *= norm / torch.norm(v[i])
    return v


def trim_full(v, undetected_byz):
    """
    Full-knowledge Trim attack.
    :param v: lista de gradientes
    :param undetected_byz: lista de dispositivos byzantinos
    """
    vi_shape = v[0].shape
    todos_grads = torch.cat(v, dim=1)
    maximum_dim = torch.max(todos_grads, dim=1).values.view(vi_shape)
    minimum_dim = torch.min(todos_grads, dim=1).values.view(vi_shape)

    # Dirección do gradiente: positivo (1) ou negativo (-1) segundo a maioría dos valores de cada param
    direction = torch.sign(torch.sum(torch.cat(v, dim=1), dim=-1, keepdim=True))

    # Se a dirección é positiva, o gradiente será o mínimo, se é negativa, o máximo
    directed_dim = (direction > 0) * minimum_dim + (direction < 0) * maximum_dim

    # # APLICAR TÉCNICA DE ATAQUE AOS DISPOSITIVOS COMPROMETIDOS
    for i in undetected_byz:
        random_12 = (1 + torch.rand(*vi_shape))
        multip = (direction * directed_dim > 0) / random_12 + (direction * directed_dim < 0) * random_12
        v[i] = directed_dim * multip
    return v


###########################################################################################
# KRUM
###########################################################################################

def matriz_distancias(v):
    norm_v = torch.sum(v ** 2, dim=0)
    dist_matriz = norm_v.view(-1, 1) + norm_v - 2 * torch.mm(v.t(), v)
    dist_matriz = torch.sqrt(torch.relu(dist_matriz))
    return dist_matriz


def computar_lambda(dist_matrix, v, c):
    n = len(v)
    n_benign = n - c
    d = len(v[0].reshape(-1))
    sqrt_d = math.sqrt(d)
    v_tensor = torch.cat(v, dim=1)

    benign_distances = dist_matrix[c:, c:]
    sum_min_distances = torch.zeros(n_benign)
    for i in range(n_benign):
        sorted_distances = torch.sort(benign_distances[i])[0]
        sum_min_distances[i] = sorted_distances[1:n-c-1].sum()

    min_sum_distances = sum_min_distances.min()
    max_re = torch.max(torch.norm(v_tensor[c:], dim=1))

    # Aplicar fórmula do teorema
    lambda_upper = (1 / ((len(v) - 2 * c - 1) * sqrt_d) * min_sum_distances) + (1 / sqrt_d * max_re)
    return lambda_upper.item()


def score(gradient, v, f):
    num_neighbours = int(v.shape[1] - 2 - f)
    benign_v = v[:, f:]
    sorted_distance, _ = torch.sort(torch.sum((benign_v - gradient) ** 2, dim=0))
    return torch.sum(sorted_distance[1:(1 + num_neighbours)]).item()


def krum(v, f):
    v_tran = torch.cat(v, dim=1)
    scores = torch.tensor([score(gradient, v_tran, f) for gradient in v])
    min_idx = int(scores.argmin(dim=0).item())
    krum_nd = v[min_idx].view(-1, 1)
    return min_idx, krum_nd


def krum_partial(v, undetected_byz_index):
    """
    Partial-knowledge Krum attack.
    :param v: lista de gradientes
    :param undetected_byz_index: lista de dispositivos byzantinos
    """
    vi_shape = v[0].shape
    lamda = 1.0

    # Seleccionar os dispositivos byzantinos, transpoñelos e calcular a dirección orixinal promediándoos
    v_tran = torch.transpose(torch.cat(v, dim=1), 0, 1)[undetected_byz_index].clone()
    original_dir = torch.mean(v_tran, dim=0).reshape(vi_shape)
    v_attack_number = 1

    # Por cada dispositivo byzantino, engádese un tensor de ataque simulado
    while v_attack_number < len(undetected_byz_index):
        lamda = 1e-2

        # Copia do tensor cos gradientes dos dispositivos byzantinos
        v_simulation = []
        for index in undetected_byz_index:
            v_simulation.append(v[index].clone())

        # Añade tensores de ataque simulados
        for i in range(v_attack_number):
            v_simulation.append(-lamda * torch.sign(original_dir))

        # Calcula o índice con menor puntuación
        min_idx, _ = krum(v_simulation, v_attack_number)

        stop_threshold = 0.00002
        while min_idx in undetected_byz_index and lamda > stop_threshold:
            lamda = lamda / 2

            not_byz_index = np.setdiff1d(np.arange(len(v_simulation)), undetected_byz_index)
            for i in range(v_attack_number):
                v_simulation[not_byz_index[i]] = -lamda * torch.sign(original_dir)

            min_idx, _ = krum(v_simulation, v_attack_number)

        v_attack_number += 1
        if min_idx not in undetected_byz_index:
            break

    for i in undetected_byz_index:
        v[i] = -lamda * torch.sign(original_dir)
    return v


def krum_full(v, undetected_byz_index):
    """
    Partial-knowledge Krum attack.
    :param v: lista de gradientes
    :param undetected_byz_index: lista de dispositivos byzantinos
    """
    stop_threshold = 1e-5
    v_tran = torch.cat(v, dim=1)
    dist_matrix = matriz_distancias(v_tran)
    min_idx, original_dir = krum(v, len(undetected_byz_index))
    original_dir = original_dir.reshape(v[0].shape)

    lamda = computar_lambda(dist_matrix, v, len(undetected_byz_index))

    while min_idx not in undetected_byz_index and lamda > stop_threshold:
        for i in undetected_byz_index:
            v[i] = - lamda * torch.sign(original_dir)
        min_idx, _ = krum(v, len(undetected_byz_index))
        lamda = lamda / 2

    for i in undetected_byz_index:
        v[i] = - lamda * torch.sign(original_dir)
    return v
