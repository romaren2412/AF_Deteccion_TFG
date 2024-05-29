import torch
import torch.nn.functional as F


def compute_trust_scores_and_normalize(client_grad_updates, server_grad_update):
    trust_scores = []
    normalized_grad_updates = []

    # Extraer y aplanar los gradientes del modelo del servidor en un solo vector
    server_grads = torch.cat([server_grad_update[key].flatten() for key in server_grad_update])
    server_norm = torch.norm(server_grads)

    for grad_update in client_grad_updates:
        # Extraer y aplanar los gradientes propuestos de cada cliente
        client_grads = torch.cat([grad_update[key].flatten() for key in grad_update])
        original_shapes = [grad_update[key].shape for key in grad_update]

        # COMPUTAR TRUST SCORES
        # Uso de cosine_similarity para medir la similitud de dirección entre los gradientes del cliente y del servidor
        cosine_sim = F.cosine_similarity(client_grads.unsqueeze(0), server_grads.unsqueeze(0), dim=1)
        trust_score = F.relu(cosine_sim)
        trust_scores.append(trust_score.item())

        # NORMALIZAR MAGNITUDES
        client_norm = torch.norm(client_grads)
        if client_norm > 0:
            normalized_flat_grad = (server_norm / client_norm) * client_grads
        else:
            normalized_flat_grad = torch.zeros_like(client_grads)

        # Reshape de los gradientes normalizados a sus formas originales
        idx = 0
        reshaped_grads = []
        for shape in original_shapes:
            num_elements = torch.prod(torch.tensor(shape)).item()
            reshaped_grad = normalized_flat_grad[idx:idx + num_elements].reshape(shape)
            reshaped_grads.append(reshaped_grad)
            idx += num_elements

        normalized_grad_updates.append(reshaped_grads)

    return trust_scores, normalized_grad_updates
