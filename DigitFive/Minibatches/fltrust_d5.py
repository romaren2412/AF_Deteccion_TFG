import torch.nn as nn
import os
import datetime

from DigitFive.Minibatches.methods import *
from rede import DigitFiveNet
from aggregation import *
from DigitFive.arquivos import *
from calculos_FLTrust import *


def fltrust_mnist(c, total_clients, byz_workers):
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
    path = os.path.join('PROBAS/', c.home_path, f"{c.data_type}_{c.extra_data_type}", timestamp)
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
        root_train, root_test = c.seleccionar_indices_root(num_data=len(rango[0]), num_extra=0)
        root_train = np.repeat(root_train[0], len(rango[0]))
        for i in range(len(total_clients)):
            print(f"[INFO USER {i}] Loading data...")
            ap = DigitFiveTraining(c, rango[i], rango_test[i])
            if i in byz_workers:
                ap.byz = True
            aprendedores.append(ap)

        # CREAR MODELO SERVIDOR
        server_model = DigitFiveTraining(c, root_train, root_test)
        testloader_global = server_model.getCreateTest()

        ########################################################################################################

        precision_array = []
        local_precisions = []
        trust_scores_array = []

        # ##################################################################################################################
        print("COMEZO DO ADESTRAMENTO...")
        # CADA ÉPOCA
        for e in range(c.EPOCH):
            client_updates = []
            client_dict_updates = []
            local_precisions_ep = []

            # CADA CLIENTE
            for i, ap in enumerate(aprendedores):
                update, update_dict = ap.sl.adestrar(nn.CrossEntropyLoss(), global_net)
                client_updates.append(update)
                client_dict_updates.append(update_dict)
                if (e + 1) % 10 == 0:
                    acc = ap.sl.test(ap.net, ap.testloader)
                    print(f"[Epoca {e}] Cliente: ", str(i), " - Accuracy: ", {acc})
                    local_precisions_ep.append(acc)

            # ADESTRAR SERVIDOR
            server_model_update, _ = server_model.sl.adestrar(nn.CrossEntropyLoss(), global_net)

            # ACTUALIZAR MODELO GLOBAL
            trust_scores, norm_updates = compute_trust_scores_and_normalize(client_updates, server_model_update)
            trust_scores_array.append(trust_scores)

            # Federar
            update_model_with_weighted_gradients(global_net, norm_updates, trust_scores, c.GLOBAL_LR)

            gardar_puntuacions(trust_scores_array, path)

            #############################################################################
            # PRECISIÓNS
            # CALCULAR A PRECISIÓN DO ENTRENO CADA 10 ITERACIÓNS
            if (e + 1) % 2 == 0:
                testear_precisions(aprendedores, testloader_global, global_net, device, e, precision_array, path,
                                   c.data_type, c.extra_data_type)
                local_precisions.append(local_precisions_ep)

        resumo_final(testloader_global, global_net, device)
    return None
