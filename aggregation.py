import torch
import csv


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
    elif t == 'bulyan':
        return bulyan(param_list, net, lr, b)
    elif t == 'multikrum':
        return multikrum(param_list, net, lr, b)
    elif t == 'multibulyan':
        return multibulyan(param_list, net, lr, b)
    elif t == 'multibulyan_var':
        return multibulyan2(param_list, net, lr, b)
    else:
        raise NotImplementedError


###########################################################################################


def aggregate(vector, net, lr):
    idx = 0
    with torch.no_grad():
        for param in net.parameters():
            if param.requires_grad:
                param.data = param.data - lr * vector[idx:(idx + param.data.numel())].reshape(param.data.shape)
                idx += param.data.numel()


def simple_mean(param_list, net, lr):
    # CÁLCULO DA MEDIA DOS GRADIENTES
    mean_nd = torch.mean(torch.cat(param_list, dim=1), dim=-1, keepdim=True)
    aggregate(mean_nd, net, lr)
    return mean_nd


def median(param_list, net, lr):
    # CÁLCULO DA MEDIANA
    median_nd = torch.median(torch.cat(param_list, dim=1), dim=-1, keepdim=True).values
    aggregate(median_nd, net, lr)
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

    # 2. Podar
    n = len(param_list)  # número de dispositivos
    m = n - len(b) * 2  # 𝑛 − 2𝑘 values restantes (𝑘 valores máximos e mínimos eliminados)

    # 3. media dos valores restantes 𝑛 − 2𝑘
    # Dentro dos dispositivos benignos, quédanse cos valores entre b e b+m
    trim_tensor = torch.mean(sorted_tensor[:, len(b):len(b) + m], dim=-1, keepdim=True)
    aggregate(trim_tensor, net, lr)

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

    # Actualizar o modelo global
    aggregate(krum_tensor, net, lr)
    return krum_tensor


def krum_gardando_seleccion(param_list, net, lr, b, path, epoch):

    # ATOPAR O MELLOR GRADIENTE
    q = len(b)  # número de dispositivos byzantinos
    # Calcular a distancia de cada gradiente ao resto dos gradientes
    v = torch.cat(param_list, dim=1)
    scores = torch.tensor([score(gradient, v, q) for gradient in param_list])

    # Escoller o gradiente coa menor distancia
    min_idx = int(scores.argmin(dim=0).item())
    krum_tensor = param_list[min_idx].view(-1, 1)

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

    # Actualizar o modelo global
    aggregate(krum_tensor, net, lr)
    return krum_tensor


def bulyan(param_list, net, lr, b):
    # Comprobar que o número de byzantinos non supera o límite
    if len(b) > (len(param_list) - 3) / 4:
        print("WARNING: Bulyan está deseñado para defender como máximo ante (n-3)/4 byzantinos.")
        print("Neste caso: ", len(b), " > ", (len(param_list) - 3) / 4)
        raise ValueError

    m = len(b)
    n = len(param_list)
    selection_set = []
    index_list = list(range(n))  # Lista de índices para referenciar las contribuciones originales

    while len(selection_set) < n - 2 * m:
        # Filtrar las contribuciones que aún no han sido seleccionadas
        current_indices = [i for i in index_list if i not in selection_set]
        current_params = [param_list[i] for i in current_indices]

        # Aplicar la función de agregación a las contribuciones filtradas
        scores = torch.tensor([score(gradient, torch.cat(current_params, dim=1), m) for gradient in current_params])
        selected_idx = int(scores.argmin(dim=0).item())

        # Añadir el índice original del parámetro seleccionado al conjunto de selección
        selection_set.append(current_indices[selected_idx])

    # Recolectar los parámetros seleccionados
    selected_params = [param_list[i] for i in selection_set]
    trim(selected_params, net, lr, b)


def multikrum(param_list, net, lr, b):
    n = len(param_list)
    f = len(b)
    m = n - 2 * f  # Número de representantes KRUM seleccionados

    # Concatenar todos los gradientes en una sola matriz para cálculos
    v = torch.cat(param_list, dim=1)

    # Calcular puntuaciones usando la función score existente
    scores = torch.tensor([score(param_list[i].view(-1, 1), v, f) for i in range(n)])

    # Seleccionar los índices de las m menores puntuaciones
    _, selected_indices = torch.topk(scores, m, largest=False, sorted=False)

    # Agregar los parámetros seleccionados
    selected_params = torch.cat([param_list[idx].view(-1, 1) for idx in selected_indices], dim=1)
    mean_params = torch.mean(selected_params, dim=1, keepdim=True)

    # Actualizar el modelo global
    aggregate(mean_params, net, lr)

    return mean_params


def multibulyan(param_list, net, lr, b):
    if len(b) > (len(param_list) - 3) / 4:
        print("WARNING: Bulyan está deseñado para defender como máximo ante (n-3)/4 byzantinos.")
        print("Neste caso: ", len(b), " > ", (len(param_list) - 3) / 4)
        raise ValueError

    # MULTIKRUM
    n = len(param_list)
    f = len(b)
    m = n - 2 * f  # Número de representantes KRUM seleccionados
    v = torch.cat(param_list, dim=1)
    scores = torch.tensor([score(param_list[i].view(-1, 1), v, f) for i in range(n)])
    _, selected_indices = torch.topk(scores, m, largest=False, sorted=False)
    selected_params = torch.cat([param_list[idx].view(-1, 1) for idx in selected_indices], dim=1)

    # BULYAN (aplicar método robusto de agregación)
    # Calcular la mediana coordenada por coordenada
    stacked_params = torch.stack([param.view(-1) for param in selected_params], dim=1)
    median_values = torch.median(stacked_params, dim=0).values

    # Calcular el promedio de los m - 2f valores más cercanos a la mediana para cada coordenada
    final_gradients = torch.zeros_like(median_values)
    for i in range(median_values.numel()):
        distances = torch.abs(stacked_params[:, i] - median_values[i])
        _, closest_indices = torch.topk(distances, m - 2 * f, largest=False, sorted=True)
        closest_values = stacked_params[closest_indices, i]
        final_gradients[i] = torch.mean(closest_values)

    # Actualizar el modelo global
    aggregate(final_gradients.view(-1, 1), net, lr)


def multibulyan2(param_list, net, lr, b):
    if len(b) > (len(param_list) - 3) / 4:
        print("WARNING: Bulyan está deseñado para defender como máximo ante (n-3)/4 byzantinos.")
        print("Neste caso: ", len(b), " > ", (len(param_list) - 3) / 4)
        raise ValueError

    # MULTIKRUM
    n = len(param_list)
    f = len(b)
    m = n - 2 * f  # Número de representantes KRUM seleccionados
    v = torch.cat(param_list, dim=1)
    scores = torch.tensor([score(param_list[i].view(-1, 1), v, f) for i in range(n)])
    _, selected_indices = torch.topk(scores, m, largest=False, sorted=False)
    selected_params = torch.cat([param_list[idx].view(-1, 1) for idx in selected_indices], dim=1)

    # BULYAN (aplicar método robusto de agregación)
    # Calcular la mediana coordenada por coordenada
    stacked_params = torch.stack([param.view(-1) for param in selected_params], dim=1)
    median_values = torch.median(stacked_params, dim=0).values

    aggregate(median_values.view(-1, 1), net, lr)
