import numpy as np
import torch


def select_byzantine_v2(t, models, undetected_byz):
    # decide attack type
    if t == 'partial_trim':
        return partial_trim(models, undetected_byz)
    elif t == 'full_trim':
        return full_trim(models, undetected_byz)
    else:
        return no_byz(models)

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


def backdoor_sen_pixel(data, target):
    inputs, labels = data
    met = len(inputs) // 2
    inputs = inputs[:met].repeat(2, 1, 1, 1)
    labels = labels[:met].repeat(2)
    for example_id in range(0, inputs.shape[0], 2):
        labels[example_id] = target
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


def edge(data, target):
    inputs, labels = data
    indices_digit = np.where(labels == target)[0]
    images_digit = inputs[indices_digit]
    qrt = len(images_digit) // 3
    inputs = np.concatenate((inputs[:qrt], images_digit[:]), axis=0)
    labels = np.concatenate((labels[:qrt], labels[indices_digit[:]]), axis=0)
    return inputs, labels


def label_flip(data):
    inputs, labels = data
    labels = (labels + 1) % 9
    return inputs, labels


#######################################################################################################################
# UNTARGETED #


def no_byz(v):
    return v


def partial_trim(models, undetected_byz):
    """
    Partial-knowledge Trim attack applied to a list of model state_dicts.
    :param models: lista de state_dicts de los modelos de todos los clientes.
    :param undetected_byz: lista de índices de los clientes byzantinos no detectados.
    """
    # Calcular las estadísticas usando los gradientes de los modelos byzantinos
    byz_models = [models[i] for i in undetected_byz]

    for name in models[0].keys():  # Asumimos que todos los modelos tienen la misma estructura
        # Concatenar los datos del mismo parámetro de todos los modelos byzantinos
        all_grads = torch.stack([model[name].data.flatten() for model in byz_models], dim=1)

        # Calcular media y desviación estándar de estos gradientes
        e_mu = torch.mean(all_grads, dim=1)
        e_sigma = torch.sqrt(torch.sum((all_grads - e_mu.view(-1, 1)) ** 2, dim=1) / len(undetected_byz))

        # Aplicar la técnica de ataque a los dispositivos comprometidos
        attack_value = (e_mu - e_sigma * torch.sign(e_mu) * 3.5).view(models[0][name].data.shape)

        for i in undetected_byz:
            models[i][name].data = attack_value

    return models


def full_trim(models, undetected_byz):
    """
    Full-knowledge Trim attack applied to a list of model state_dicts.
    :param models: lista de state_dicts de los modelos de todos los clientes.
    :param undetected_byz: lista de índices de los clientes byzantinos no detectados.
    """
    for name in models[0].keys():  # Asumimos que todos los modelos tienen la misma estructura
        all_grads = torch.stack([model[name].data.flatten() for model in models], dim=1)

        # Calcula el máximo y mínimo para cada dimensión de los gradientes
        maximum_dim = torch.max(all_grads, dim=1).values
        minimum_dim = torch.min(all_grads, dim=1).values

        # Determina la dirección general de los gradientes
        direction = torch.sign(torch.sum(all_grads, dim=1))

        # Selecciona el gradiente dirigido según la dirección
        directed_dim = (direction > 0) * minimum_dim + (direction < 0) * maximum_dim
        directed_dim = directed_dim.view(models[0][name].data.shape)

        # Aplicar la técnica de ataque
        random_12 = 2  # Este valor debería ser ajustado o calculado dinámicamente
        for i in undetected_byz:
            directed_value = directed_dim * (
                        (direction.view(models[0][name].data.shape) * directed_dim > 0) / random_12 +
                        (direction.view(models[0][name].data.shape) * directed_dim < 0) * random_12)
            models[i][name].data = directed_value

    return models


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


def mean_attack(v):
    """
    :param v: vector de gradientes
    :return:
    """
    for i in range(len(v)):
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


def full_mean_attack_v2(model, undetected_byz_index, grad_list):
    """
    :param model: modelo
    :param undetected_byz_index: lista de dispositivos byzantinos
    :param grad_list: lista de gradientes
    :return:
    """
    # Se todos os dispositivos restantes están comprometidos, invértese o signo dos gradientes do modelo
    if len(undetected_byz_index) == len(grad_list):
        for param in model.parameters():
            param.grad.data = -param.grad.data
        return model

    # Se hai parte dos dispositivos comprometidos, invértese o signo dos gradientes dos mesmos
    vi_shape = grad_list[0].shape
    todos_grads = torch.cat(grad_list, dim=1)
    grad_sum = torch.sum(todos_grads, dim=1)

    # Para obter o array de clientes benignos (e calcular a suma do gradiente)
    benign_clients = np.setdiff1d(np.arange(todos_grads.size(1)), undetected_byz_index)
    benign_grad_sum = torch.sum(todos_grads[:, benign_clients], dim=1)

    # Invírtese a un tipo de media no que ten máis importancia o gradiente dos dispositivos benignos
    new_val = ((-grad_sum + benign_grad_sum) / len(undetected_byz_index)).reshape(vi_shape)
    model.parameters.grad = new_val
    return model


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
