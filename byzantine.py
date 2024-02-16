import numpy as np
import torch


def no_byz_range(v, f):
    return v


def label_flip_range(each_worker_label, undetected_byz_index):
    # 1. LABEL FLIP: cambiar as etiquetas asinadas ás mostras de datos dos traballadores byzantinos.
    """
    :param each_worker_label: lista coas etiquetas obxectivo de cada traballador
    :param undetected_byz_index: lista de clientes byzantinos (índices respecto aos n clientes que haxa nese momento)
    :return: each_worker_label actualizado
    """
    for i in undetected_byz_index:
        each_worker_label[i] = (each_worker_label[i] + 1) % 9
    return each_worker_label


def backdoor_range(each_worker_data, each_worker_label, undetected_byz, target_backdoor_dba):
    # Igual que backdoor_range en MLR pero sin redimensionar a (-1,784)
    # 2. BACKDOOR: introducir un patrón específico nas mostras de datos para engañar ao modelo central.
    """
    :param each_worker_data:
    :param each_worker_label:
    :param undetected_byz: lista de clientes byzantinos
    :param target_backdoor_dba: ETIQUETA DO PATRÓN
    :return: each_worker_data e each_worker_label actualizados
    """
    for i in undetected_byz:
        # Duplica as primeiras 300 mostras de datos de cada traballador byzantino
        # REDIMENSIONAR OS DATOS PARA O BUCLE
        prim = each_worker_data[i][:300]  # {tensor(300, 784)}
        sec = prim.view(-1, 1, 28, 28)  # {tensor(300, 1, 28, 28)}
        each_worker_data[i] = sec.repeat(2, 1, 1, 1)  # {tensor(600, 1, 28, 28)}
        each_worker_label[i] = each_worker_label[i][:300].repeat(2)

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


def edge_range(each_worker_data, each_worker_label, undetected_byz, test_edge_images, label):
    """
    # 3. EDGE: introduce mostras específicas do número 7 (label 1) nas mostras de datos dos traballadores byzantinos.
    :param each_worker_data:
    :param each_worker_label:
    :param undetected_byz: lista de clientes byzantinos
    :param test_edge_images:
    :param label:
    :return: each_worker_data e each_worker_label actualizados
    """
    for i in undetected_byz:
        each_worker_data[i] = torch.cat((each_worker_data[i][:150], test_edge_images[:450]), dim=0)
        each_worker_label[i] = torch.cat((each_worker_label[i][:150], label[:450]), dim=0)

    return each_worker_data, each_worker_label


def dba_range(each_worker_data, each_worker_label, undetected_byz, target_backdoor_dba):
    # IGUAL QUE DBA_RANGE en MLR PERO SIN REDIMENSIONAR A (-1,784)

    subarrays = np.array_split(undetected_byz, 4)
    for index, subarray in enumerate(subarrays):
        # Grupo 1: Establece o patrón 1 na posición (26, 26)
        if index == 0:
            for i in subarray:
                each_worker_data[i] = each_worker_data[i][:300].view(-1, 1, 28, 28).repeat(2, 1, 1, 1)
                each_worker_label[i] = each_worker_label[i][:300].repeat(2)
                for example_id in range(0, each_worker_data[i].shape[0], 2):
                    each_worker_data[i][example_id][0][26][26] = 1.
                    each_worker_label[i][example_id] = target_backdoor_dba

        # Grupo 2: Establece o patrón 1 na posición (24, 26)
        elif index == 1:
            for i in subarray:
                each_worker_data[i] = each_worker_data[i][:300].view(-1, 1, 28, 28).repeat(2, 1, 1, 1)
                each_worker_label[i] = each_worker_label[i][:300].repeat(2)
                for example_id in range(0, each_worker_data[i].shape[0], 2):
                    each_worker_data[i][example_id][0][24][26] = 1.
                    each_worker_label[i][example_id] = target_backdoor_dba

        # Grupo 3: Establece o patrón 1 na posición (26, 24)
        elif index == 2:
            for i in subarray:
                each_worker_data[i] = each_worker_data[i][:300].view(-1, 1, 28, 28).repeat(2, 1, 1, 1)
                each_worker_label[i] = each_worker_label[i][:300].repeat(2)
                for example_id in range(0, each_worker_data[i].shape[0], 2):
                    each_worker_data[i][example_id][0][26][24] = 1.
                    each_worker_label[i][example_id] = target_backdoor_dba

        # Grupo 4: Establece o patrón 1 na posición (25, 25)
        else:
            for i in subarray:
                each_worker_data[i] = each_worker_data[i][:300].view(-1, 1, 28, 28).repeat(2, 1, 1, 1)
                each_worker_label[i] = each_worker_label[i][:300].repeat(2)
                for example_id in range(0, each_worker_data[i].shape[0], 2):
                    each_worker_data[i][example_id][0][25][25] = 1.
                    each_worker_label[i][example_id] = target_backdoor_dba

    return each_worker_data, each_worker_label


def partial_trim_range(v, undetected_byz):
    """
    Partial-knowledge Trim attack. w.l.o.g., Asumimos que os primeiros f traballadores están comprometidos.
    :param v: lista de gradientes
    :param undetected_byz: lista de dispositivos byzantinos
    """
    # first compute the statistics
    vi_shape = v[0].shape
    todos_grads = torch.cat(v, dim=1)
    byz_grads = todos_grads[:, undetected_byz]
    e_mu = torch.mean(byz_grads, dim=1)  # mean
    # standard deviation
    e_sigma = torch.sqrt(torch.sum((byz_grads - e_mu.view(-1, 1)) ** 2, dim=1) / len(undetected_byz))

    # APLICAR TÉCNICA NOS DISPOSITIVOS COMPROMETIDOS
    for i in undetected_byz:
        v[i] = (e_mu - e_sigma * torch.sign(e_mu) * 3.5).view(vi_shape)

    return v


def full_trim_range(v, undetected_byz):
    """
    v: lista de gradientes
    undetected_byz: lista de dispositivos byzantinos
    """
    vi_shape = v[0].shape
    todos_grads = torch.cat(v, dim=1)
    maximum_dim = torch.max(todos_grads, dim=1).values.view(vi_shape)
    minimum_dim = torch.min(todos_grads, dim=1).values.view(vi_shape)

    # Dirección do gradiente: positivo (1) ou negativo (-1) segundo a maioría dos valores
    direction = torch.sign(torch.sum(torch.cat(v, dim=1), dim=-1, keepdim=True))

    # Se a dirección é positiva, o gradiente será o mínimo, se é negativa, o máximo
    directed_dim = (direction > 0) * minimum_dim + (direction < 0) * maximum_dim

    # # APLICAR TÉCNICA DE ATAQUE AOS DISPOSITIVOS COMPROMETIDOS
    # Se a dirección e o vector teñen a mesma orientación, o gradiente redúcese
    # Se a dirección e o vector teñen orientacións opostas, o gradiente aumenta
    random_12 = 2
    for i in undetected_byz:
        v[i] = directed_dim * ((direction * directed_dim > 0) / random_12 + (direction * directed_dim < 0) * random_12)
    return v


def gaussian_attack_range(v, undetected_byz):
    """
    :param v: lista de gradientes
    :param undetected_byz: lista de dispositivos byzantinos
    :return:
    """
    # A forma do primeiro gradiente é a forma de cada gradiente
    vi_shape = v[0].shape

    adv_grads = torch.cat(v, dim=1)  # Concatenamos os gradientes
    e_mu = torch.mean(adv_grads, dim=1)  # Media

    # Desviación estándar
    e_sigma = torch.sqrt(torch.sum((adv_grads - e_mu.view(-1, 1)) ** 2, dim=1) / len(undetected_byz))

    # APLICAR TÉCNICA DE ATAQUE NOS DISPOSITIVOS COMPROMETIDOS
    for i in undetected_byz:
        # 1. Calcula a norma do gradiente do dispositivo comprometido
        norm = torch.norm(v[i])
        # 2. Xérar un vector aleatorio con media e_mu e desviación estándar e_sigma
        v[i] = torch.normal(e_mu, e_sigma).view(vi_shape)
        # 3. Normalizar o gradiente adversario para que teña a mesma norma que o gradiente orixinal
        v[i] *= norm / torch.norm(v[i])
    return v


def mean_attack_range(v, undetected_byz):
    """
    :param v: vector de gradientes
    :param undetected_byz: lista de clientes byzantinos
    :return:
    """
    for i in undetected_byz:
        v[i] = -v[i]
    return v


def full_mean_attack_range(v, undetected_byz_index):
    """
    :param v: vector de gradientes
    :param undetected_byz_index: lista de clientes byzantinos (indices respecto aos clientes totais)
    :return:
    """
    # Se todos os dispositivos están comprometidos, invértese o signo de todos os gradientes
    if len(undetected_byz_index) == len(v):
        for i in undetected_byz_index:
            v[i] = -v[i]
        return v

    # Se non hai dispositivos comprometidos, non se fai nada
    if len(undetected_byz_index) == 0:
        return v

    # Se hai parte dos dispositivos comprometidos, invértese o signo dos gradientes dos mesmos
    vi_shape = v[0].shape
    todos_grads = torch.cat(v, dim=1)
    grad_sum = torch.sum(todos_grads, dim=1)

    # Para obter o array de clientes benignos (e calcular a suma do gradiente)
    benign_clients = np.setdiff1d(np.arange(todos_grads.size(1)), undetected_byz_index)

    benign_grad_sum = torch.sum(todos_grads[:, benign_clients], dim=1)

    # Invírtese a un tipo de media no que ten máis importancia o gradiente dos dispositivos benignos
    new_val = ((-grad_sum + benign_grad_sum) / len(undetected_byz_index)).reshape(vi_shape)
    for i in undetected_byz_index:
        v[i] = new_val
    return v


"""
SCORE (para KRUM):
A función score calcula a distancia entre un gradiente dado e os gradientes dos dispositivos
"""


def score(gradient, v, f):
    """
    :param gradient:
    :param v:
    :param f: número de dispositivos byzantinos
    :return:
    """
    # Número de veciños: n - 2 - k
    num_neighbours = int(v.shape[1] - 2 - f)
    # Ordena os gradientes segundo a distancia ao gradiente dado
    sorted_distance, _ = torch.sort(torch.sum((v - gradient) ** 2, dim=0))
    # Devolve a suma das distancias dos veciños (exclúe o propio gradiente)
    return torch.sum(sorted_distance[1:(1 + num_neighbours)]).item()


def krum(v, f):
    """
    :param v: gradientes simulados
    :param f: número de dispositivos byzantinos
    :return:
    """
    # Se o número de dispositivos é menor que o número de veciños, non se pode aplicar Krum
    if len(v) - f - 2 <= 0:
        f = len(v) - 3

    # Se os gradientes teñen máis dunha dimensión, cóncatanse nun tensor
    if len(v[0].shape) > 1:
        v_tran = torch.cat(v, dim=1)
    else:
        v_tran = v

    # Calcula a puntuación de cada gradiente
    # scores: puntuacións de "similitude" dos demais gradientes co gradiente dado
    scores = torch.tensor([score(gradient, v_tran, f) for gradient in v])

    # Devolve o índice do gradiente con menor puntuación e o seu valor
    min_idx = scores.argmin().item()
    krum_nd = v[min_idx].reshape(-1)
    return min_idx, krum_nd


def dir_partial_krum_lambda_range(v, undetected_byz_index):
    """
    :param v: parámetros da rede
    :param undetected_byz_index: lista de dispositivos byzantinos
    :return:
    """
    vi_shape = v[0].shape
    lamda = 1.0

    # Seleccionar os dispositivos byzantinos, transpoñelos e calcular a dirección orixinal promediándoos
    v_tran = torch.transpose(torch.cat(v, dim=1), 0, 1)[undetected_byz_index].clone()
    original_dir = torch.mean(v_tran, dim=0).reshape(vi_shape)
    v_attack_number = 1

    # Por cada dispositivo byzantino, engádese un tensor de ataque simulado
    while v_attack_number < len(undetected_byz_index):
        lamda = 1.0

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

            # Ataca os dispositivos "non byzantinos"?
            not_byz_index = np.setdiff1d(np.arange(len(v_simulation)), undetected_byz_index)
            for i in range(v_attack_number):
                v_simulation[not_byz_index[i]] = -lamda * torch.sign(original_dir)

            min_idx, _ = krum(v_simulation, v_attack_number)

        v_attack_number += 1

        # Se o min_idx non é un dispositivo byzantino, rompe o bucle
        if min_idx not in undetected_byz_index:
            break

    print('chosen lambda:', lamda)
    # v[0] = -lamda * torch.sign(original_dir)
    for i in undetected_byz_index:
        # random_raw = torch.rand(vi_shape) - 0.5
        # random_norm = torch.rand(1).item() * epsilon
        # randomness = random_raw * random_norm / torch.norm(random_raw)
        v[i] = -lamda * torch.sign(original_dir)

    return v


def dir_full_krum_lambda_range(v, undetected_byz_index, epsilon=0.01):
    """
    :param v:
    :param undetected_byz_index:
    :param epsilon:
    :return:
    """
    # Se hai menos de 2 dispositivos, seleccionase un aleatorio
    # Neste caso, o ataque máis forte é cambiar o signo do gradiente
    if len(v) <= 2:
        for i in undetected_byz_index:
            v[i] = -v[i]
        return v

    # Se hai máis de 2 dispositivos, aplícase Krum
    vi_shape = v[0].shape
    _, original_dir = krum(v, len(undetected_byz_index))
    original_dir = original_dir.reshape(vi_shape)

    lamda = 1.
    for i in undetected_byz_index:
        v[i] = -lamda * torch.sign(original_dir)
    min_idx, _ = krum(v, len(undetected_byz_index))
    stop_threshold = 1e-5

    while min_idx not in undetected_byz_index and lamda > stop_threshold:
        lamda = lamda / 2
        for i in undetected_byz_index:
            v[i] = -lamda * torch.sign(original_dir)
        min_idx, _ = krum(v, len(undetected_byz_index))

    print('chosen lambda:', lamda)
    v[0] = -lamda * torch.sign(original_dir)
    for i in range(1, len(undetected_byz_index)):
        index = undetected_byz_index[i]
        random_raw = torch.rand(vi_shape) - 0.5
        random_norm = torch.rand(1).item() * epsilon
        randomness = random_raw * random_norm / torch.norm(random_raw)
        v[index] = -lamda * torch.sign(original_dir) + randomness

    return v


def scaling_attack_range(v, undetected_byz):
    scaling_factor = len(v)
    for param_id in undetected_byz:
        v[param_id] = v[param_id] * scaling_factor
    return v
