import datetime
import os

import numpy as np
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader

from MNIST.arquivos import *
from MNIST.datos import repartir_datos, preparar_datos, crear_dataset_auxiliar
from MNIST.methods import inicializar_global_model, create_local_models
from aggregation import update_model_with_weighted_gradients
from calculos_FLARE import *


def flare(c, total_clients, byz_workers):
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

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join('MNIST/PROBAS/', c.home_path, timestamp, c.byz_type)
    if not os.path.exists(path):
        os.makedirs(path)

    # EJECUCIÓN
    with device:
        ########################################################################################################
        # CARGA DO DATASET
        train_data, test_data = preparar_datos()
        global_test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=500, shuffle=False)
        worker_loaders = repartir_datos(c, train_data, len(total_clients))

        # DATASET AUXILIAR
        aux_dataset = crear_dataset_auxiliar()
        aux_loader = DataLoader(aux_dataset, batch_size=c.BATCH_SIZE, shuffle=True,
                                generator=torch.Generator(device='cuda'))
        ####################################################################################################

        # ARQUITECTURA DO MODELO - CNN
        global_net = inicializar_global_model(device, aux_loader, c.LR)
        aprendedores = create_local_models(len(total_clients), c, worker_loaders, test_data, byz_workers, global_net)

        ####################################################################################################

        # set upt parameters
        ben_workers = [i for i in total_clients if i not in byz_workers]
        print("CLIENTES BENIGNOS: ", ben_workers)
        print("CLIENTES BYZANTINOS: ", byz_workers)
        print("----------------------------------")

        precision_array = []
        local_precision_array_ep = []
        local_precisions = []
        trust_scores_array = []
        local_epoch = 0

        ###################################################################################################

        # EXECUTAR ATAQUES
        target_backdoor_dba = 7
        if c.byz_type == 'dba':
            for index, g in enumerate(np.array_split(byz_workers, 4)):
                for byzantine in g:
                    aprendedores[byzantine].dba_index = index

        # ##################################################################################################################
        print("COMEZO DO ADESTRAMENTO...")
        # CADA ÉPOCA
        for e in range(c.EPOCH):
            all_updates = []
            all_plrs = []

            # ADESTRAMENTO DE CADA CLIENTE
            for i, ap in enumerate(aprendedores):
                update = ap.sl.adestrar(c, nn.CrossEntropyLoss(), global_net, target_backdoor_dba)
                plr = extraer_plrs(ap.net, aux_loader, device)
                # EXTRACCIÓN DE PLRs
                all_updates.append(update)
                all_plrs.append(plr)
                if (e + 1) % 5 == 0:
                    acc = ap.sl.test(ap.net, ap.testloader)
                    print(f"[Epoca {e}, {local_epoch}] Cliente: ", str(i), " - Accuracy: ", {acc})
                    local_precision_array_ep.append(acc)

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
            # CALCULAR A PRECISIÓN DO ENTRENO CADA 20 ITERACIÓNS
            testear_precisions(global_test_data_loader, global_net, device, e, precision_array, path,
                               target_backdoor_dba, c.byz_type)

        resumo_final(global_test_data_loader, global_net, device)
