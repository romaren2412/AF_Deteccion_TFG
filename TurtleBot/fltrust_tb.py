from redes import TurtlebotNet
from turtlebot import *
import copy
from calculos_FLTrust import *
from arquivos import *
from aggregation import *
from evaluate_tb import test_tb
import datetime


def crear_loader_auxiliar(num_samples=10):
    # Generar datos aleatorios de imagen
    images = torch.rand(num_samples, 1440)

    return images, None


def fltrust_tb(c, total_clients):
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

    path = os.path.join('PROBAS', c.home_path, f"4_{c.tipo_ben}", c.tipo_mal)
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
            ap.net = TurtlebotNet().to(device)
            ap.create_train_test(i)
            aprendedores.append(ap)

        # MODELO SERVIDOR
        c.RANK = len(total_clients)
        server_model = TurtlebotTraining(c, server=True)
        server_model.net = TurtlebotNet().to(device)
        server_model.create_train_test(i)
        testloader_global = server_model.testloader

        ####################################################################################################

        # Definir la pérdida de entropía cruzada softmax
        criterion = aprendedores[-1].criterion
        trust_scores_array = []
        precision_array = []
        local_precisions = []

        # ##################################################################################################################
        print("COMEZO DO ADESTRAMENTO...")
        # CADA ÉPOCA
        for e in range(c.EPOCH_tb):
            client_updates = []
            client_dicts = []
            local_precisions_ep = []
            # CADA CLIENTE
            for i, ap in enumerate(aprendedores):
                update, dicts = ap.sl.adestrar_tb(ap.criterion, global_net)
                client_updates.append(update)
                client_dicts.append(dicts)
                """
                if (e + 1) % 20 == 0:
                    _, _, acc = test_tb(ap.testloader, ap.net, ap.criterion, ap.device)
                    print(f"[Epoca {e}] LOCAL || Cliente: ", str(i), " - Accuracy: ", {acc})
                    local_precisions_ep.append(acc)
                    _, _, acc = test_tb(testloader_global, ap.net, ap.criterion, ap.device)
                    print(f"[Epoca {e}] GLOBAL || Cliente: ", str(i), " - Accuracy: ", {acc})
                """

            # ADESTRAR SERVIDOR
            server_model_update, _ = server_model.sl.adestrar_tb(criterion, global_net)

            # ACTUALIZAR MODELO GLOBAL
            trust_scores, norm_updates = compute_trust_scores_and_normalize(client_updates, server_model_update)
            trust_scores_array.append(trust_scores)

            update_model_with_weighted_gradients(global_net, norm_updates, trust_scores, c.LR_tb)
            # update_model_with_equal_gradients(global_net, norm_updates, c.LR_tb)

            if (e + 1) % 10 == 0:
                testear_precisions(testloader_global, global_net, device, e, precision_array, path, criterion)

            """
            if (e + 1) % 20 == 0:
                local_precisions.append(local_precisions_ep)
                gardar_precisions_locais(path, local_precisions, [0])
            """

        resumo_final(testloader_global, global_net, device, e, trust_scores_array, path, criterion)
    return None
