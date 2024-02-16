import torch

def lbfgs(s_k_list, y_k_list, v):
    # Concatenar S_k_list e Y_k_list_tensor
    curr_S_k = torch.cat(s_k_list, dim=1)
    curr_Y_k = torch.cat(y_k_list, dim=1)

    # Calcular r_k, l_k, sigma_k, d_k_diag, upper_mat, lower_mat, mat, mat_inv
    s_k_time_y_k = torch.matmul(curr_S_k.t(), curr_Y_k)
    s_k_time_s_k = torch.matmul(curr_S_k.t(), curr_S_k)
    r_k = torch.triu(torch.from_numpy(s_k_time_y_k.cpu().numpy()))
    r_k_tensor = r_k.to(curr_S_k.device)
    l_k = s_k_time_y_k - r_k_tensor
    sigma_k = torch.matmul(y_k_list[-1].T, s_k_list[-1]) / torch.matmul(s_k_list[-1].T, s_k_list[-1])
    d_k_diag = torch.diag(s_k_time_y_k)
    upper_mat = torch.cat([sigma_k * s_k_time_s_k, l_k], dim=1)
    lower_mat = torch.cat([l_k.T, -torch.diag(d_k_diag)], dim=1)
    mat = torch.cat([upper_mat, lower_mat], dim=0)
    mat_inv = torch.inverse(mat)

    approx_prod = sigma_k * v
    p_mat = torch.cat([torch.matmul(curr_S_k.t(), sigma_k * v), torch.matmul(curr_Y_k.t(), v)], dim=0)
    approx_prod -= torch.matmul(torch.matmul(torch.cat([sigma_k * curr_S_k, curr_Y_k], dim=1), mat_inv), p_mat)

    return approx_prod


def lbfgsFedRec(delta_w, delta_g, v):
    # Concatenar S_k_list e Y_k_list_tensor
    curr_W_k = torch.cat(delta_w, dim=1)
    curr_G_k = torch.cat(delta_g, dim=1)

    A = torch.matmul(curr_W_k.T, curr_G_k)
    D = torch.diag_embed(torch.diag(A))
    L = torch.tril(A, diagonal=-1)
    sigma = torch.matmul(delta_g[-1].T, delta_w[-1]) / torch.matmul(delta_w[-1].T, delta_w[-1])

    upper_mat = torch.cat([-D, L.T], dim=1)
    lower_mat = torch.cat([L, torch.matmul((sigma * curr_W_k.t()), curr_W_k)], dim=1)
    mat = torch.cat([upper_mat, lower_mat], dim=0)
    mat1 = torch.inverse(mat)
    mat2 = torch.cat([torch.matmul(curr_G_k.T, v), torch.matmul(sigma * curr_W_k.T, v)], dim=0)
    p = torch.matmul(mat1, mat2)

    Hv = sigma * v - torch.matmul(torch.cat([curr_G_k, sigma * curr_W_k], dim=1), p)

    return Hv


"""
def lbfgs2(w_k_list, g_k_list, v):
    # Concatenar S_k_list e Y_k_list_tensor
    curr_w_k = torch.cat(w_k_list, dim=1)
    curr_g_k = torch.cat(g_k_list, dim=1)

    # Calcular R_k, L_k, sigma_k, D_k_diag, upper_mat, lower_mat, mat, mat_inv
    W_k_time_G_k = torch.matmul(curr_w_k.t(), curr_g_k)
    W_k_time_W_k = torch.matmul(curr_w_k.t(), curr_w_k)
    D_k_diag = torch.diag(W_k_time_G_k)
    L_k = torch.tril(W_k_time_G_k, diagonal=-1)
    sigma_k = torch.matmul(g_k_list[-1].T, w_k_list[-1]) / torch.matmul(w_k_list[-1].T, w_k_list[-1])

    # Cholesky genera nan's
    mat_cholesky = sigma_k * W_k_time_W_k + torch.matmul(torch.matmul(L_k, torch.diag(D_k_diag)), L_k.T)
    Jt = torch.linalg.cholesky(mat_cholesky)

    # Na diagonal hai valores negativos, as raíces cuadradas son nan's
    Dt_elev1 = torch.diag(torch.sqrt(D_k_diag))
    Dt_elev2 = torch.diag(D_k_diag ** (-1 / 2))

    upper_m1 = torch.cat([-1 * Dt_elev1, torch.matmul(Dt_elev2, L_k.T)], dim=1)
    lower_m1 = torch.cat([torch.zeros_like(Jt.T), Jt.T], dim=1)
    mat1 = torch.inverse(torch.cat([upper_m1, lower_m1], dim=0))

    upper_m2 = torch.cat([Dt_elev1, torch.zeros_like(Dt_elev1)], dim=1)
    lower_m2 = torch.cat([torch.matmul(Dt_elev2, L_k.T), Jt], dim=1)
    mat2 = torch.inverse(torch.cat([upper_m2, lower_m2], dim=0))

    mat3 = torch.cat([torch.matmul(curr_g_k.T, v), torch.matmul(sigma_k * curr_w_k.T, v)], dim=0)

    q = torch.matmul(torch.matmul(mat1, mat2), mat3)

    return sigma_k * v - torch.matmul(torch.cat([curr_g_k, sigma_k * curr_w_k], dim=1), q)
"""