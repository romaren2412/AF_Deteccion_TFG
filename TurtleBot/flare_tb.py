# coding: utf-8
import copy
import datetime

from aggregation import *
from arquivos import *
from calculos_FLARE import *
from rede import TurtlebotNet
from turtlebot import *


def crear_loader_auxiliar(num_samples=10):
    # Generar datos aleatorios de imagen
    images = torch.rand(num_samples, 1440)

    return images, None


def flare_tb(c, total_clients):
    """
    Detecta ataques mediante clustering.
    :param c: obxecto de configuración
    :param total_clients: lista cos clientes totais (a partir do segundo adestramento, os supostamente benignos)
    :return:
    """
    # Decide el dispositivo de ejecución
    if c.gpu == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', c.gpu)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join('PROBAS', c.home_path, f"4_{c.tipo_ben}", c.tipo_mal, timestamp)
    if not os.path.exists(path):
        os.makedirs(path)

    # EJECUCIÓN
    with device:
        # ARQUITECTURA DO MODELO - CNN
        global_net = TurtlebotNet()

        # Mueve el modelo a la GPU o CPU según el contexto
        global_net.to(device)

        ########################################################################################################
        # CARGA DO DATASET
        aprendedores = []
        for i in range(len(total_clients)):
            c_ = copy.deepcopy(c)
            c_.RANK = i

            print(f"[INFO USER {i}] Loading data...")
            ap = TurtlebotTraining(c_)
            # CREAR REDE, TRAIN E TEST
            ap.net = copy.deepcopy(global_net).float().to(ap.device)
            ap.create_train_test(i)

            if i == 0:
                global_test_temp = ap.init_global_test()
            elif i == len(total_clients) - 1:
                global_test_temp = ap.create_global_test(global_test_temp)
            else:
                ap.continue_global_test(global_test_temp)

            aprendedores.append(ap)
        testloader_global = global_test_temp

        # DATASET AUXILIAR
        aux_loader = crear_loader_auxiliar()

        ####################################################################################################

        # Definir la pérdida de entropía cruzada softmax
        criterion = aprendedores[-1].criterion

        precision_array = []
        local_precision_array_ep = []
        local_precisions = []
        trust_scores_array = []

        # ##################################################################################################################
        print("COMEZO DO ADESTRAMENTO...")
        # CADA ÉPOCA
        for e in range(c.EPOCH_tb):
            all_updates = []
            all_plrs = []
            # CADA CLIENTE
            for ap in aprendedores:
                update = ap.sl.adestrar_tb(ap.criterion, global_net)
                plr = extraer_plrs_tb(ap.net, aux_loader, device)
                # EXTRACCIÓN DE PLRs
                all_updates.append(update)
                all_plrs.append(plr)

            # Calcular MMD
            mmd_matrix = crear_matriz_mmd(all_plrs)
            nearest_neighbors_counts = select_top_neighbors(mmd_matrix, len(aprendedores))
            trust_scores = softmax(nearest_neighbors_counts, temperature=1.0)
            trust_scores_array.append(trust_scores.tolist())

            # Federar
            update_model_with_weighted_gradients(global_net, [list(u.values()) for u in all_updates], trust_scores)

            # Gardar resultados
            gardar_puntuacions(trust_scores_array, path, [0])
            local_precisions.append(local_precision_array_ep)
            gardar_precisions_locais(path, local_precisions, [0])

            local_precision_array_ep = []
            #############################################################################
            # PRECISIÓNS
            # CALCULAR A PRECISIÓN DO ENTRENO CADA 10 ITERACIÓNS
            if (e + 1) % 10 == 0:
                testear_precisions(testloader_global, global_net, device, e, precision_array, path, criterion)

        resumo_final(testloader_global, global_net, device, e, trust_scores_array, path, criterion)
    return None
