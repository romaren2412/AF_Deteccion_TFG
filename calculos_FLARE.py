import torch


def gaussian_kernel(a, b, sigma=1.0):
    sq_dist = torch.cdist(a, b).pow(2)
    return torch.exp(-sq_dist / (2 * sigma ** 2))


def mmd_loss(x, y, sigma=1.0):
    K_xx = gaussian_kernel(x, x, sigma)
    K_yy = gaussian_kernel(y, y, sigma)
    K_xy = gaussian_kernel(x, y, sigma)

    return K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()


def crear_matriz_mmd(plrs):
    n = len(plrs)
    mmd_matrix = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            mmd_matrix[i, j] = mmd_matrix[j, i] = mmd_loss(plrs[i], plrs[j])
    return mmd_matrix


def select_top_neighbors(mmd_matrix, n):
    num_neighbors = n // 2  # 50% de n
    nearest_neighbors_counts = torch.zeros(n)
    for i in range(n):
        mmd_scores = mmd_matrix[i].clone()
        mmd_scores[i] = float('inf')

        # Obtiene los índices de los vecinos más cercanos (menores valores de MMD)
        neighbors = torch.topk(mmd_scores, k=num_neighbors, largest=False).indices
        nearest_neighbors_counts[neighbors] += 1
    return nearest_neighbors_counts


def softmax(x, temperature=1.0):
    e_x = torch.exp((x - x.max()) / temperature)
    return e_x / e_x.sum()


def extraer_plrs(model, data_loader, device):
    model.eval()
    plr = None
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            plr = model.extraer_plr(inputs)

    return plr.detach()
