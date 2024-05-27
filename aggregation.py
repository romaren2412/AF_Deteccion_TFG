from copy import deepcopy

import torch


def aggregate_models(model_updates, trust_scores):
    # Normalizar trust_scores
    trust_scores = [ts / sum(trust_scores) for ts in trust_scores]
    aggregated_model = deepcopy(model_updates[0])
    for key in aggregated_model.keys():
        aggregated_model[key] = aggregated_model[key] * trust_scores[0]
        for i in range(1, len(model_updates)):
            aggregated_model[key] += model_updates[i][key] * trust_scores[i]
    return aggregated_model


def equal_aggregate_models(model_updates):
    trust_scores = [1 / len(model_updates) for _ in range(len(model_updates))]
    aggregated_model = deepcopy(model_updates[0])
    for key in aggregated_model.keys():
        aggregated_model[key] = aggregated_model[key] * trust_scores[0]
        for i in range(1, len(model_updates)):
            aggregated_model[key] += model_updates[i][key] * trust_scores[i]
    return aggregated_model


def aggregate_updates(global_model, normalized_updates, trust_scores):
    aggregated_gradients = [torch.zeros_like(grad) for grad in normalized_updates[0]]

    # Agregar cada grupo de gradientes ponderado por su respectivo puntaje de confianza
    for update, score in zip(normalized_updates, trust_scores):
        for i, grad in enumerate(update):
            aggregated_gradients[i] += (score * grad) / sum(trust_scores)

    # Aplicar los gradientes agregados al modelo global
    with torch.no_grad():
        for param, agg_grad in zip(global_model.parameters(), aggregated_gradients):
            param.data -= agg_grad  # Asignar directamente los gradientes actualizados (descent gradient)

    return global_model


def update_model_with_weighted_gradients(net, normalized_grad_updates, trust_scores, lr):
    with torch.no_grad():
        total_trust = sum(trust_scores)  # Suma total de los puntajes de confianza
        if total_trust == 0:
            raise ValueError("La suma total de los trust scores no puede ser cero.")

        # Inicializar la actualización ponderada global como cero
        weighted_gradient_sum = [torch.zeros_like(param) for param in net.parameters()]

        # Ponderar los gradientes por los trust scores y sumarlos a la suma ponderada global
        for normalized_grads, score in zip(normalized_grad_updates, trust_scores):
            weight = score / total_trust
            for param, norm_grad, sum_grad in zip(net.parameters(), normalized_grads, weighted_gradient_sum):
                sum_grad += norm_grad * weight

        # Aplicar la actualización al modelo global
        for param, weighted_grad in zip(net.parameters(), weighted_gradient_sum):
            if param.requires_grad:
                param.data -= lr * weighted_grad


def equal_update(net, client_gradients, lr):
    client_gradients_flatten = [torch.cat([param.flatten() for param in grad]) for grad in client_gradients]
    mean_nd = torch.mean(torch.stack(client_gradients_flatten), dim=0)
    with torch.no_grad():
        idx = 0
        # Actualización dos parámetros
        for j, param in enumerate(net.parameters()):
            if param.requires_grad:
                param.data = param.data - lr * mean_nd[idx:(idx + param.data.numel())].reshape(param.data.shape)
                idx = idx + param.data.numel()
