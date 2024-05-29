import torch


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
                param.data += lr * weighted_grad


def update_model_with_equal_gradients(net, normalized_grad_updates, lr):
    equal_pond = 1 / len(normalized_grad_updates)
    with torch.no_grad():
        # Inicializar la actualización ponderada global como cero
        weighted_gradient_sum = [torch.zeros_like(param) for param in net.parameters()]

        # Sumar los gradientes ponderados globalmente
        for normalized_grads in normalized_grad_updates:
            for norm_grad, sum_grad in zip(normalized_grads, weighted_gradient_sum):
                sum_grad += norm_grad * equal_pond

        # Aplicar la actualización al modelo global
        for param, weighted_grad in zip(net.parameters(), weighted_gradient_sum):
            if param.requires_grad:
                param.data += lr * weighted_grad

