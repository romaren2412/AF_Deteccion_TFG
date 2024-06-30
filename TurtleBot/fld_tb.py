import datetime

from clases_redes import TurtlebotNet
from lbfgs import *
from detection import *
from turtlebot import *
import copy
from np_aggregation import *
from arquivos import *


def fl_detector(c, args, total_clients, entrenamento, original_clients, byz_workers):
    """
    Detecta ataques mediante clustering.
    :param c: obxecto de configuración
    :param args: obxecto cos argumentos de entrada
    :param total_clients: lista cos clientes totais (a partir do segundo adestramento, os supostamente benignos)
    :param entrenamento: número de adestramento (para gardar os resultados no caso de repetir o adestramento desde 0)
    :param original_clients: lista cos clientes orixinais (para gardar os resultados no caso de repetir o adestramento)
    :param byz_workers: lista cos clientes byzantinos
    :return:
    """
    # Decide el dispositivo de ejecución
    if args.gpu == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.gpu)

    # Crear ou abrir ficheiro para gardar os resultados
    para_string = "nepochs: " + str(c.EPOCH_tb) + ", lr: " + str(
        c.LR_tb) + ", batch_size: " + str(c.BACH_SIZE_tb) + ", nworkers: " + str(
        c.SIZE) + ", nbyz: " + str(len(byz_workers))

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join('../PROBAS/PROBAS', args.home_path, f"4_{args.tipo_ben}", args.tipo_mal)
    if not os.path.exists(path):
        os.makedirs(path)
    ataques_path = path + '/Ataques_Detectados.txt'
    with (open(ataques_path, 'w+')) as f:
        f.write(para_string + '\n')

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

            if i in byz_workers:
                ap.byz = True

            aprendedores.append(ap)
            testloader_global = global_test_temp

        ####################################################################################################

        # Definir la pérdida de entropía cruzada softmax
        criterion = aprendedores[-1].criterion

        # set upt parameters
        undetected_byz = np.intersect1d(byz_workers, total_clients)
        undetected_byz_index = np.where(np.isin(total_clients, undetected_byz))[0]

        ben_workers = [i for i in original_clients if i not in byz_workers]
        # Gardar os clientes orixinais se é o primeiro adestramento
        if entrenamento == 0:
            print("CLIENTES ORIXINAIS: ", original_clients)
            print("CLIENTES BENIGNOS: ", ben_workers)
            print("CLIENTES BYZANTINOS: ", byz_workers)
            print("----------------------------------")
            with open(ataques_path, 'w+') as f:
                f.write("CLIENTES ORIXINAIS: " + str(original_clients) + '\n')
                f.write("CLIENTES BENIGNOS: " + str(ben_workers) + '\n')
                f.write("CLIENTES BYZANTINOS: " + str(byz_workers) + '\n')
                f.write("----------------------------------\n")

        print("----------------------------------")
        print("ADESTRAMENTO: ", entrenamento)
        temp_ben_workers = [i for i in total_clients if i not in byz_workers]
        percent_ben_orixinal = len(temp_ben_workers) / len(ben_workers)
        print("Benignos: ", temp_ben_workers)
        print("Byzantinos (non detectados aínda): ", undetected_byz.tolist())
        print(" --> Porcentaxe de benignos restantes: ", percent_ben_orixinal.__format__('0.2%'))

        with open(ataques_path, 'w+') as f:
            f.write("ADESTRAMENTO: " + str(entrenamento) + '\n')
            f.write("Benignos: " + str(temp_ben_workers) + '\n')
            f.write("Byzantinos (non detectados aínda): " + str(undetected_byz.tolist()) + '\n')
            f.write(" --> Porcentaxe de benignos restantes: " + str(percent_ben_orixinal) + '\n')

        epochs = c.EPOCH_tb
        lr = c.LR_tb
        grad_list = []
        old_grad_list = []
        weight_record = []
        grad_record = []
        malicious_score = []
        precision_array = []

        # ##################################################################################################################
        print("COMEZO DO ADESTRAMENTO...")
        # CADA ÉPOCA
        for e in range(epochs):
            # print("EPOCH: ", e)
            # CADA CLIENTE
            for ap in aprendedores:
                ap.net.load_state_dict(global_net.state_dict())
                ap.sl.adestrar_tb(ap.criterion, grad_list)

            # param_list: Lista de tensores cos gradientes aplanados dos parámetros de cada cliente
            param_list = [torch.cat([xx.view(-1, 1) for xx in x], dim=0) for x in grad_list]
            # tmp: Copia temporal dos parámetros do modelo actual
            tmp = [param.data.clone() for param in global_net.parameters()]
            # weight: Lista de tensores cos parámetros aplanados do modelo actual
            weight = torch.cat([x.view(-1, 1) for x in tmp], dim=0)

            # CALCULAR HESSIAN VECTOR PRODUCT CON LBFGS (A PARTIR DA EPOCA 50)
            if e >= c.FLDET_START:
                hvp = lbfgs(weight_record, grad_record, weight - last_weight)
            else:
                hvp = None

            # SELECCIONAR MÉTODO DE AGREGACIÓN
            grad, distance = select_aggregation(args.aggregation, old_grad_list, param_list, global_net, lr,
                                                undetected_byz_index, hvp)

            # ACTUALIZAR A DISTANCIA MALICIOSA
            if distance is not None and e > c.FLDET_START:
                malicious_score.append(distance)

            # DETECCION DE CLIENTES MALICIOSOS
            if len(malicious_score) > 10 and args.tipo_exec != 'no_detect':
                mal_scores = np.sum(malicious_score[-10:], axis=0)
                det = detectar_maliciosos(mal_scores, para_string, e, total_clients, undetected_byz_index, path)
                if det is not None:
                    datos_finais(path, precision_array, testloader_global, global_net, device, e, malicious_score,
                                 criterion)
                    return det

            # ACTUALIZAR O PESO E O GRADIENTE
            if e > 0:
                weight_record.append(weight - last_weight)
                grad_record.append(grad - last_grad)

            # LIMPAR MEMORIA E REINICIAR A LISTA
            if len(weight_record) > 10:
                del weight_record[0]
                del grad_record[0]

            # ACTUALIZAR PARÁMETROS
            last_weight = weight
            last_grad = grad
            old_grad_list = param_list
            del grad_list
            grad_list = []

            #############################################################################
            # PRECISIÓNS
            # CALCULAR A PRECISIÓN DO ENTRENO CADA 10 ITERACIÓNS
            if (e + 1) % 50 == 0:
                testear_precisions(testloader_global, global_net, device, e, precision_array, path, criterion)

            # GARDAR AS PUNTUACIÓNS DO ENTRENO CADA 10 ITERACIÓNS
            if (e + 1) % 10 == 0 and e > c.FLDET_START:
                gardar_puntuacions(malicious_score, path)

        resumo_final(testloader_global, global_net, device, e, malicious_score, path, criterion)
    return None
