import os

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from aggregation import *
from arquivos import *
from calculos_FLARE import *
from methods import *
from redes import DigitFiveNet


def crear_dataset_auxiliar(num_samples=10, img_size=(16, 16), num_classes=10):
    # Generar datos_d5.txt aleatorios de imagen
    images = torch.rand(num_samples, 3, img_size[0], img_size[1])

    # Generar etiquetas aleatorias
    labels = torch.randint(low=0, high=num_classes, size=(num_samples,))

    # Crear un TensorDataset
    dataset = TensorDataset(images, labels)

    return dataset


def flare_d5(c, total_clients, byz_workers):
    """
    Detecta ataques mediante clustering.
    :param c: obxecto de configuración
    :param total_clients: lista cos clientes totais (a partir do segundo adestramento, os supostamente benignos)
    :param byz_workers: lista cos clientes byzantinos
    :return:
    """
    # Decide el dispositivo de ejecución
    if c.gpu == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', c.gpu)

    path = os.path.join('PROBAS_REF/', c.home_path, f"{c.data_type}+/{c.data_type}_{c.extra_data_type}")
    if not os.path.exists(path):
        os.makedirs(path)

    # EJECUCIÓN
    with device:
        # INICIALIZAR MODELO GLOBAL
        global_net = DigitFiveNet()
        global_net.initialize_weights()
        global_net.to(device)

        # APRENDEDORES
        # CARGA DO DATASET
        aprendedores = []
        rango, rango_test = c.create_ranks()
        _, root_test = c.seleccionar_indices_root(num_data=len(rango[0]), num_extra=0)
        for i in range(len(total_clients)):
            print(f"[INFO USER {i}] Loading data...")
            ap = DigitFiveTraining(c, rango[i], rango_test[i])
            if i in byz_workers:
                ap.byz = True
            aprendedores.append(ap)

        # DATASET AUXILIAR
        aux_dataset = crear_dataset_auxiliar()
        aux_loader = DataLoader(aux_dataset, batch_size=c.BATCH_SIZE, shuffle=True,
                                generator=torch.Generator(device='cuda'))

        global_test_data_loader = aprendedores[0].create_global_test(root_test)

        ########################################################################################################

        precision_array = []
        local_precision_array_ep = []
        local_precisions = []
        trust_scores_array = []
        local_epoch = 0

        # ##################################################################################################################
        print("COMEZO DO ADESTRAMENTO...")
        # CADA ÉPOCA
        for e in range(c.EPOCH):
            all_updates = []
            all_plrs = []

            # CADA CLIENTE
            for i, ap in enumerate(aprendedores):
                update = ap.sl.adestrar(c, nn.CrossEntropyLoss(), global_net)
                plr = extraer_plrs(ap.net, aux_loader, device)
                # EXTRACCIÓN DE PLRs
                all_updates.append(update)
                all_plrs.append(plr)
                """
                if (e + 1) % (c.EPOCH // 3) == 0 or (e + 1) == c.EPOCH:
                    acc = ap.sl.test(ap.net, ap.testloader)
                    print(f"[Epoca {e}, {local_epoch}] Cliente: ", str(i), " - Accuracy: ", {acc})
                    local_precision_array_ep.append(acc)
                """

            # Calcular MMD
            mmd_matrix = crear_matriz_mmd(all_plrs)
            nearest_neighbors_counts = select_top_neighbors(mmd_matrix, len(aprendedores))
            trust_scores = softmax(nearest_neighbors_counts, temperature=1.0)
            trust_scores_array.append(trust_scores.tolist())

            # Federar
            update_model_with_weighted_gradients(global_net, [list(u.values()) for u in all_updates], trust_scores)

            # Gardar resultados
            gardar_puntuacions(trust_scores_array, path, byz_workers)
            local_precisions.append(local_precision_array_ep)
            gardar_precisions_locais(path, local_precisions, byz_workers)

            local_epoch = 0
            local_precision_array_ep = []

            #############################################################################
            # PRECISIÓNS
            # CALCULAR A PRECISIÓN DO ENTRENO 10 veces
            if (e + 1) % (c.EPOCH // 10) == 0 or (e + 1) == c.EPOCH:
                testear_precisions(aprendedores, global_test_data_loader, global_net, device, e, precision_array, path,
                                   c.data_type, c.extra_data_type)

        resumo_final(global_test_data_loader, global_net, device)
    return None
