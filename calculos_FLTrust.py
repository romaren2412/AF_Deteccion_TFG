import torch
import torch.nn.functional as F


def compute_trust_scores(client_updates, server_update):
    trust_scores = []
    # Extraer los parámetros del modelo del servidor y aplanarlos en un solo vector
    server_weights = torch.cat([val.flatten() for val in server_update.values()])

    for update in client_updates:
        # Extraer y aplanar los parámetros del modelo del cliente en un solo vector
        client_weights = torch.cat([val.flatten() for val in update.values()])
        # Calcular la similitud del coseno entre los pesos del cliente y del servidor
        cosine_sim = F.cosine_similarity(client_weights.unsqueeze(0), server_weights.unsqueeze(0), dim=1)
        # Aplicar ReLU para obtener la puntuación de confianza
        trust_score = F.relu(cosine_sim)
        trust_scores.append(trust_score)

    # Concatenar todas las puntuaciones de confianza en un tensor para manipulación futura
    return torch.cat(trust_scores)


def compute_trust_scores_and_normalize(client_grad_updates, server_grad_update):
    trust_scores = []
    normalized_grad_updates = []

    # Extraer y aplanar los gradientes del modelo del servidor en un solo vector
    server_grads = torch.cat([g.flatten() for g in server_grad_update])
    server_norm = torch.norm(server_grads)

    for grad_update in client_grad_updates:
        # Extraer y aplanar los gradientes propuestos de cada cliente
        client_grads = torch.cat([g.flatten() for g in grad_update])
        original_shapes = [g.shape for g in grad_update]

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


def reconstruct_and_load_state_dict(model, flat_state_dict):
    state_dict = {}
    index = 0
    for key, val in model.state_dict().items():
        state_dict[key] = flat_state_dict[index:index + val.numel()].reshape(val.shape)
        index += val.numel()
    model.load_state_dict(state_dict)
