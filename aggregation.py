import torch


def update_model_with_weighted_gradients(net, updates, trust_scores):
    with torch.no_grad():
        # Inicializar la actualización ponderada global como cero
        weighted_gradient_sum = [torch.zeros_like(param) for param in net.parameters()]

        # Ponderar los gradientes por los trust scores y sumarlos a la suma ponderada global
        for update, score in zip(updates, trust_scores):
            for param, up, sum_grad in zip(net.parameters(), update, weighted_gradient_sum):
                sum_grad += up * score

        # Aplicar la actualización al modelo global
        for param, weighted_grad in zip(net.parameters(), weighted_gradient_sum):
            if param.requires_grad:
                param.data += weighted_grad
