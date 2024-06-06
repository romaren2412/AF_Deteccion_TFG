import torch


###########################################################################################
def select_aggregation(t, param_list, net, lr, b):
    # decide aggregation type
    if t == 'simple_mean':
        return simple_mean(param_list, net, lr)
    elif t == 'trim':
        return trim(param_list, net, lr, b)
    elif t == 'krum':
        return krum(param_list, net, lr, b)
    elif t == 'median':
        return median(param_list, net, lr)
    else:
        raise NotImplementedError


###########################################################################################


def simple_mean(param_list, net, lr):
    # CÁLCULO DA MEDIA DOS GRADIENTES
    mean_nd = torch.mean(torch.cat(param_list, dim=1), dim=-1, keepdim=True)

    with torch.no_grad():
        idx = 0
        # Actualización dos parámetros
        for j, param in enumerate(net.parameters()):
            if param.requires_grad:
                param.data = param.data - lr * mean_nd[idx:(idx + param.data.numel())].reshape(param.data.shape)
                idx = idx + param.data.numel()

    return mean_nd


def median(param_list, net, lr):
    # CÁLCULO DA MEDIANA
    median_nd = torch.median(torch.cat(param_list, dim=1), dim=-1, keepdim=True).values

    # Actualizar modelo global
    idx = 0
    for param in net.parameters():
        if param.requires_grad:
            param.data = param.data - lr * median_nd[idx:(idx + param.data.numel())].reshape(param.data.shape)
            idx += param.data.numel()

    return median_nd


###########################################################################################

"""
Trimmed-Mean first sorts the values of the corresponding coordinates in the clients’ model updates.
After removing the largest and the smallest 𝑘 values, Trimmed-Mean calculates the average of the
remaining 𝑛 − 2𝑘 values
"""


def trim(param_list, net, lr, b):
    # Trimmed-Mean calcula a media dos valores restantes 𝑛 − 2𝑘, despois de eliminar os 𝑘 valores máximos e mínimos
    # 1. Ordenar
    sorted_tensor, indices = torch.sort(torch.cat(param_list, dim=1), dim=-1)

    # Contar la influencia de los clientes bizantinos
    byz_podados = (indices[:, :len(b)] < len(b)).sum(dim=-1) + (indices[:, -len(b):] < len(b)).sum(dim=-1)
    media_podados = byz_podados.float().mean().item() / len(b)
    print(f"Media de bizantinos podados por parámetro: {media_podados:.2%}")

    # 2. Podar
    n = len(param_list)  # número de dispositivos
    m = n - len(b) * 2  # 𝑛 − 2𝑘 values restantes (𝑘 valores máximos e mínimos eliminados)

    # 3. media dos valores restantes 𝑛 − 2𝑘
    # Dentro dos dispositivos benignos, quédanse cos valores entre b e b+m
    trim_tensor = torch.mean(sorted_tensor[:, len(b):len(b) + m], dim=-1, keepdim=True)

    # Actualizar modelo global
    idx = 0
    for param in net.parameters():
        if param.requires_grad:
            param.data = param.data - lr * trim_tensor[idx:(idx + param.data.numel())].reshape(param.data.shape)
            idx += param.data.numel()

    return trim_tensor


"""
Krum tries to find a single model update among the clients’ model updates
as the aggregated model update in each iteration.
The chosen model update is the one with the closest
Euclidean distances to the nearest 𝑛 − 𝑘 − 2 model updates.
"""


def score(gradient, v, f):
    num_neighbours = v.shape[1] - 2 - f
    sorted_distance, _ = torch.sort(torch.sum((v - gradient) ** 2, dim=0))
    return torch.sum(sorted_distance[1:(1 + num_neighbours)]).item()


def krum(param_list, net, lr, b):

    # ATOPAR O MELLOR GRADIENTE
    q = len(b)  # número de dispositivos byzantinos
    # Calcular a distancia de cada gradiente ao resto dos gradientes
    v = torch.cat(param_list, dim=1)
    scores = torch.tensor([score(gradient, v, q) for gradient in param_list])

    # Escoller o gradiente coa menor distancia
    min_idx = int(scores.argmin(dim=0).item())
    krum_tensor = param_list[min_idx].view(-1, 1)

    """
    Guardar las puntuaciones en un archivo CSV
    with open(path + "/score_krum.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        if epoch == 0:  # Añadir encabezados en la primera iteración
            headers = [f'Cliente {i + 1}' for i in range(len(scores))]
            writer.writerow(['Iteracion'] + headers)
        writer.writerow([epoch] + scores.tolist())

    # Escoller o gradiente coa menor distancia
    min_idx = int(scores.argmin(dim=0).item())
    krum_tensor = param_list[min_idx].view(-1, 1)

    with open(path + '/selected_krum.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if epoch == 0:
            writer.writerow(['Iteracion', 'Gradiente seleccionado'])
        writer.writerow([epoch, min_idx])

    # Imprimir si o gradiente escollido é byzantino
    if min_idx in b:
        print(f"O gradiente escollido é byzantino: {min_idx}")
    """
    # Actualizar o modelo global
    idx = 0
    for param in net.parameters():
        if param.requires_grad:
            param.data = param.data - lr * krum_tensor[idx:(idx + param.data.numel())].reshape(param.data.shape)
            idx += param.data.numel()

    return krum_tensor
