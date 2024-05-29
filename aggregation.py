import torch


def update_model_with_weighted_gradients(net, updates, trust_scores):
    with torch.no_grad():
        # Inicializar la actualizaci�n ponderada global como cero
        weighted_gradient_sum = [torch.zeros_like(param) for param in net.parameters()]

        # Ponderar los gradientes por los trust scores y sumarlos a la suma ponderada global
        for update, score in zip(updates, trust_scores):
            for param, up, sum_grad in zip(net.parameters(), update, weighted_gradient_sum):
                sum_grad += up * score

        # Aplicar la actualizaci�n al modelo global
        for param, weighted_grad in zip(net.parameters(), weighted_gradient_sum):
            if param.requires_grad:
                param.data += weighted_grad


def update_model_with_equal_gradients(net, updates):
    equal_pond = 1 / len(updates)
    with torch.no_grad():
        # Inicializar la actualizaci�n ponderada global como cero
        weighted_gradient_sum = [torch.zeros_like(param) for param in net.parameters()]

        # Sumar los gradientes ponderados globalmente
        for update in updates:
            for param, up, sum_grad in zip(net.parameters(), update, weighted_gradient_sum):
                sum_grad += up * equal_pond

        # Aplicar la actualizaci�n al modelo global
        for param, weighted_grad in zip(net.parameters(), weighted_gradient_sum):
            if param.requires_grad:
                param.data += weighted_grad
