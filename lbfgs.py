import torch


def lbfgs_github(s_k_list, y_k_list, v):
    # Concatenar S_k_list e Y_k_list_tensor
    curr_s_k = torch.cat(s_k_list, dim=1)
    curr_y_k = torch.cat(y_k_list, dim=1)

    # Calcular r_k, l_k, sigma_k, d_k_diag, upper_mat, lower_mat, mat, mat_inv
    s_k_time_y_k = torch.matmul(curr_s_k.t(), curr_y_k)
    s_k_time_s_k = torch.matmul(curr_s_k.t(), curr_s_k)
    r_k = torch.triu(torch.from_numpy(s_k_time_y_k.cpu().numpy()))
    r_k_tensor = r_k.to(curr_s_k.device)
    l_k = s_k_time_y_k - r_k_tensor
    sigma_k = torch.matmul(y_k_list[-1].T, s_k_list[-1]) / torch.matmul(s_k_list[-1].T, s_k_list[-1])
    d_k_diag = torch.diag(s_k_time_y_k)
    upper_mat = torch.cat([sigma_k * s_k_time_s_k, l_k], dim=1)
    lower_mat = torch.cat([l_k.T, -torch.diag(d_k_diag)], dim=1)
    mat = torch.cat([upper_mat, lower_mat], dim=0)
    mat_inv = torch.inverse(mat)

    approx_prod = sigma_k * v
    p_mat = torch.cat([torch.matmul(curr_s_k.t(), sigma_k * v), torch.matmul(curr_y_k.t(), v)], dim=0)
    approx_prod -= torch.matmul(torch.matmul(torch.cat([sigma_k * curr_s_k, curr_y_k], dim=1), mat_inv), p_mat)

    return approx_prod


def lbfgs(delta_w, delta_g, v):
    # Concatenar S_k_list e Y_k_list_tensor
    curr_w_k = torch.cat(delta_w, dim=1)
    curr_g_k = torch.cat(delta_g, dim=1)

    a = torch.matmul(curr_w_k.T, curr_g_k)
    d = torch.diag_embed(torch.diag(a))
    low = torch.tril(a, diagonal=-1)
    sigma = torch.matmul(delta_g[-1].T, delta_w[-1]) / torch.matmul(delta_w[-1].T, delta_w[-1])

    upper_mat = torch.cat([-d, low.T], dim=1)
    lower_mat = torch.cat([low, torch.matmul((sigma * curr_w_k.t()), curr_w_k)], dim=1)
    mat = torch.cat([upper_mat, lower_mat], dim=0)
    mat1 = torch.inverse(mat)
    mat2 = torch.cat([torch.matmul(curr_g_k.T, v), torch.matmul(sigma * curr_w_k.T, v)], dim=0)
    p = torch.matmul(mat1, mat2)

    return sigma * v - torch.matmul(torch.cat([curr_g_k, sigma * curr_w_k], dim=1), p)
