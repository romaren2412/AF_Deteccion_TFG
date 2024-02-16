import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def simple_mean(old_gradients, param_list, net, lr, b, hvp=None):
    """
        :param old_gradients: lista de gradientes calculados polos dispositivos
        :param param_list: lista de parámetros da rede
        :param net: modelo global
        :param lr: taxa de aprendizaxe
        :param b: lista de dispositivos byzantinos
        :param hvp: hessian-vector product
    """
    # HAI HESSIANA CALCULADA --> CÁLCULO DE DISTANCIA
    if hvp is not None:
        pred_grad = []

        # Predición dos gradientes --> gradiente vello + hvp
        for i in range(len(old_gradients)):
            pred_grad.append(old_gradients[i] + hvp)

        # Vector de predicións
        pred = np.zeros(len(old_gradients))
        # b: lista de dispositivos byzantinos --> Os b vectores son "anómalos"
        pred[b] = 1

        # Se non hai clientes byzantinos ou todos son byzantinos, calcular distancia directamente
        if len(b) == 0 or len(b) == len(old_gradients):
            # DISTANCIA ENTRE GRADIENTES PREDITOS E NOVOS
            distancia = torch.norm(torch.cat(pred_grad, dim=1) - torch.cat(param_list, dim=1), dim=0).cpu().numpy()

        # CÁLCULO DE DISTANCIAS E ÁREA BAIXO A CURVA ROC:
        # MÉTRICA PARA AVALIAR A CALIDADE DUN MODELO DE CLASIFICACIÓN BINARIA
        else:
            # DISTANCIA A: ENTRE GRADIENTES VELLOS E NOVOS
            distancia = torch.norm(torch.cat(old_gradients, dim=1) - torch.cat(param_list, dim=1), dim=0).cpu().numpy()
            auc1 = roc_auc_score(pred, distancia)

            # DISTANCIA B: DISTANCIA ENTRE GRADIENTES PREDITOS E NOVOS
            distancia = torch.norm(torch.cat(pred_grad, dim=1) - torch.cat(param_list, dim=1), dim=0).cpu().numpy()
            auc2 = roc_auc_score(pred, distancia)
            print("Detection AUC: %0.4f; Detection AUC: %0.4f" % (auc1, auc2))

        # NORMALIZACIÓN DE DISTANCIAS
        distancia = distancia / np.sum(distancia)
    else:
        distancia = None

    # CÁLCULO DA MEDIA DOS GRADIENTES
    mean_nd = torch.mean(torch.cat(param_list, dim=1), dim=-1, keepdim=True)

    idx = 0
    # Actualización dos parámetros
    for j, param in enumerate(net.parameters()):
        if param.requires_grad:
            param.data = param.data - lr * mean_nd[idx:(idx + param.data.numel())].reshape(param.data.shape)
            idx = idx + param.data.numel()

    return mean_nd, distancia


"""
Trimmed-Mean first sorts the values of the corresponding coordinates in the clients’ model updates.
After removing the largest and the smallest 𝑘 values, Trimmed-Mean calculates the average of the
remaining 𝑛 − 2𝑘 values
"""


def trim(old_gradients, param_list, net, lr, b, hvp=None):
    """
    :param old_gradients: lista de gradientes calculados polos dispositivos
    :param param_list: lista de parámetros da rede
    :param net: modelo global
    :param lr: taxa de aprendizaxe
    :param b: lista de dispositivos byzantinos
    :param hvp: hessian-vector product
    """
    if hvp is not None:
        pred_grad = []
        for i in range(len(old_gradients)):
            pred_grad.append(old_gradients[i] + hvp)
        pred = np.zeros(len(old_gradients))
        pred[b] = 1
        if len(b) == 0 or len(b) == len(old_gradients):
            distancia = torch.norm(torch.cat(pred_grad, dim=1) - torch.cat(param_list, dim=1), dim=0).cpu().numpy()
        else:
            distancia = torch.norm(torch.cat(old_gradients, dim=1) - torch.cat(param_list, dim=1), dim=0).cpu().numpy()
            auc1 = roc_auc_score(pred, distancia)
            distancia = torch.norm(torch.cat(pred_grad, dim=1) - torch.cat(param_list, dim=1), dim=0).cpu().numpy()
            auc2 = roc_auc_score(pred, distancia)
            print("Detection AUC: %0.4f; Detection AUC: %0.4f" % (auc1, auc2))
        distance = distancia / np.sum(distancia)
    else:
        distance = None

    # Trimmed-Mean calcula a media dos valores restantes 𝑛 − 2𝑘, despois de eliminar os 𝑘 valores máximos e mínimos
    # 1. Ordenar
    sorted_tensor, _ = torch.sort(torch.cat(param_list, dim=1), dim=-1)

    # 2. Podar
    n = len(param_list)  # número de dispositivos
    m = n - len(b) * 2  # 𝑛 − 2𝑘 values restantes (𝑘 valores máximos e mínimos eliminados)

    # 3. media dos valores restantes 𝑛 − 2𝑘
    # Dentro dos dispositivos benignos, quédanse cos valores entre b e b+m
    benign_workers = np.setdiff1d(np.arange(n), b)
    trim_benign = benign_workers[:m]
    trim_tensor = torch.mean(sorted_tensor[:, trim_benign], dim=-1, keepdim=True)

    # Actualizar modelo global
    idx = 0
    for param in net.parameters():
        if param.requires_grad:
            param.data = param.data - lr * trim_tensor[idx:(idx + param.data.numel())].reshape(param.data.shape)
            idx += param.data.numel()

    return trim_tensor, distance


"""
Median calculates the median value of the corresponding coordinates in all model updates and treats it as
the corresponding coordinate of the aggregated model update.
"""


def median(old_gradients, param_list, net, lr, b, hvp=None):
    """
        :param old_gradients: lista de gradientes calculados polos dispositivos
        :param param_list: lista de parámetros da rede
        :param net: modelo global
        :param lr: taxa de aprendizaxe
        :param b: número de dispositivos byzantinos
        :param hvp: hessian-vector product
        """
    if hvp is not None:
        pred_grad = []
        for i in range(len(old_gradients)):
            pred_grad.append(old_gradients[i] + hvp)
        pred = np.zeros(len(old_gradients))
        pred[b] = 1
        if len(b) == 0 or len(b) == len(old_gradients):
            distancia = torch.norm(torch.cat(pred_grad, dim=1) - torch.cat(param_list, dim=1), dim=0).cpu().numpy()
        else:
            distancia = torch.norm(torch.cat(old_gradients, dim=1) - torch.cat(param_list, dim=1), dim=0).cpu().numpy()
            auc1 = roc_auc_score(pred, distancia)
            distancia = torch.norm(torch.cat(pred_grad, dim=1) - torch.cat(param_list, dim=1), dim=0).cpu().numpy()
            auc2 = roc_auc_score(pred, distancia)
            print("Detection AUC: %0.4f; Detection AUC: %0.4f" % (auc1, auc2))
        distance = distancia / np.sum(distancia)
    else:
        distance = None

    # CÁLCULO DA MEDIANA
    if len(param_list) % 2 == 1:
        # Se hai un número impar de dispositivos, a mediana é o valor central
        median_nd = torch.sort(torch.cat(param_list, dim=1), dim=-1)[0][:, len(param_list) // 2]
        median_nd = median_nd.view(-1, 1)
    else:
        # Se hai un número par de dispositivos, a mediana é a media dos dous valores centrais
        median_nd = torch.sort(torch.cat(param_list, dim=1), dim=-1)[0][:, len(param_list) // 2: len(param_list) // 2 + 1].mean(dim=-1, keepdim=True)

    # Actualizar modelo global
    idx = 0
    for param in net.parameters():
        if param.requires_grad:
            param.data = param.data - lr * median_nd[idx:(idx + param.data.numel())].reshape(param.data.shape)
            idx += param.data.numel()

    return median_nd, distance


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


def krum(old_gradients, param_list, net, lr, b, hvp=None):
    """
        :param old_gradients: lista de gradientes calculados polos dispositivos
        :param param_list: lista de parámetros da rede
        :param net: modelo global
        :param lr: taxa de aprendizaxe
        :param b: número de dispositivos byzantinos
        :param hvp: hessian-vector product
    """
    if hvp is not None:
        pred_grad = []
        for i in range(len(old_gradients)):
            pred_grad.append(old_gradients[i] + hvp)
        pred = np.zeros(len(old_gradients))
        pred[b] = 1
        if len(b) == 0:
            distancia = torch.norm(torch.cat(pred_grad, dim=1) - torch.cat(param_list, dim=1), dim=0).cpu().numpy()
        else:
            distancia = torch.norm(torch.cat(old_gradients, dim=1) - torch.cat(param_list, dim=1), dim=0).cpu().numpy()
            auc1 = roc_auc_score(pred, distancia)
            distancia = torch.norm(torch.cat(pred_grad, dim=1) - torch.cat(param_list, dim=1), dim=0).cpu().numpy()
            auc2 = roc_auc_score(pred, distancia)
            print("Detection AUC: %0.4f; Detection AUC: %0.4f" % (auc1, auc2))
        distance = distancia / np.sum(distancia)
    else:
        distance = None

    # ATOPAR O MELLOR GRADIENTE
    num_params = len(param_list)  # número de dispositivos
    q = len(b)  # número de dispositivos byzantinos
    if num_params <= 2:
        # Se non hai suficientes clientes, escoller un de forma aleatoria
        random_idx = np.random.choice(num_params)
        krum_tensor = param_list[random_idx].view(-1, 1)
    else:
        if num_params - q - 2 <= 0:  # n-k-2 <= 0 (se non hai polo menos 3 dispositivos benignos)
            q = num_params - 3  # q = n-3    (asumir que hai n-3 dispositivos byzantinos)

        # Calcular a distancia de cada gradiente ao resto dos gradientes
        v = torch.cat(param_list, dim=1)
        scores = torch.tensor([score(gradient, v, q) for gradient in param_list])

        # Escoller o gradiente coa menor distancia
        min_idx = int(scores.argmin(dim=0).item())
        krum_tensor = param_list[min_idx].view(-1, 1)

    # Actualizar o modelo global
    idx = 0
    for param in net.parameters():
        if param.requires_grad:
            param.data = param.data - lr * krum_tensor[idx:(idx + param.data.numel())].reshape(param.data.shape)
            idx += param.data.numel()

    return krum_tensor, distance
