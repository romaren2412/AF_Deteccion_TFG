import torch.nn as nn
import datetime

import clases_redes as cr

from lbfgs import *
from byzantine import *
from detection import *
from arquivos import *
from np_aggregation import select_aggregation
from datos import *


def porcentaxes(each_worker_label):
    for i in range(len(each_worker_label)):
        print("Cliente ", i)
        for j in range(10):
            print("Porcentaxe de ", j, ": ", torch.sum(each_worker_label[i] == j).item() / len(each_worker_label[i]))
        print("Total: ", len(each_worker_label[i]))


def fl_detector(args, total_clients, entrenamento, original_clients):
    """
    Detecta ataques mediante clustering.
    :param args: obxecto cos argumentos de entrada
    :param total_clients: lista cos clientes totais (a partir do segundo adestramento, os supostamente benignos)
    :param entrenamento: número de adestramento (para gardar os resultados no caso de repetir o adestramento desde 0)
    :param original_clients: lista cos clientes orixinais (para gardar os resultados no caso de repetir o adestramento)
    :return:
    """
    # Decide el dispositivo de ejecución
    if args.gpu == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.gpu)

        # Crear ou abrir ficheiro para gardar os resultados
    para_string = "bias: " + str(args.bias) + ", nepochs: " + str(args.nepochs) + ", lr: " + str(
        args.lr) + ", nworkers: " + str(args.nworkers) + ", nbyz: " + str(args.nbyz)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join('PROBAS', args.home_path, timestamp, args.aggregation, args.byz_type, str(entrenamento))
    if not os.path.exists(path):
        os.makedirs(path)
    ataques_path = path + '/Ataques_Detectados.txt'
    with (open(ataques_path, 'w+')) as f:
        f.write(para_string + '\n')

    # EJECUCIÓN
    with device:
        num_outputs = 10

        # ARQUITECTURA DO MODELO - CNN
        net = cr.CNN_v2(num_channels=1, num_outputs=num_outputs)
        net.initialize_weights()

        # Mueve el modelo a la GPU o CPU según el contexto
        net.to(device)

        ########################################################################################################
        # CARGA DO DATASET
        train_data_loader, test_data_loader, test_data = preparar_datos()

        if args.byz_type == 'edge':
            digit_edge = 7
            indices_digit = torch.nonzero(test_data.targets == digit_edge, as_tuple=False)[:, 0]
            images_digit = test_data.data[indices_digit, :] / 255.0
            test_edge_images = images_digit.view(-1, 1, 28, 28).to(device)
            edge_label = torch.ones(len(test_edge_images)).to(torch.int).to(device)

        ####################################################################################################

        # DECIDIR EL TIPO DE ATAQUE
        byz = select_byzantine_range(args.byz_type)

        # Definir la pérdida de entropía cruzada softmax
        softmax_cross_entropy = nn.CrossEntropyLoss()

        # set upt parameters
        num_workers = len(total_clients)
        byz_workers = [i for i in range(args.nbyz)]
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

        lr = args.lr
        epochs = args.nepochs
        grad_list = []
        old_grad_list = []
        weight_record = []
        grad_record = []
        train_acc_list = []
        test_edge_images = None
        edge_label = None

        ###################################################################################################

        # ASIGNACIÓN ALEATORIA DOS DATOS ENTRE OS CLIENTES
        each_worker_data, each_worker_label = repartir_datos(args, train_data_loader, num_workers, device)
        # porcentaxes(each_worker_label)

        ######################################################################################################

        # EXECUTAR ATAQUES
        target_backdoor_dba = 0
        each_worker_data, each_worker_label = ataque_pre_entreno(args.byz_type, each_worker_data, each_worker_label,
                                                                 undetected_byz_index, target_backdoor_dba,
                                                                 test_edge_images, edge_label)

        # ##################################################################################################################
        # set malicious scores
        malicious_score = []

        # CADA ÉPOCA
        for e in range(epochs):
            # CADA CLIENTE
            for i in range(num_workers):
                inputs, labels = each_worker_data[i][:], each_worker_label[i][:]
                # NON ACUMULAR GRADIENTES
                net.zero_grad()
                outputs = net(inputs)
                loss = softmax_cross_entropy(outputs, labels)
                loss.backward()
                grad_list.append([param.grad.clone() for param in net.parameters()])

            # param_list: Lista de tensores cos gradientes aplanados dos parámetros de cada cliente
            param_list = [torch.cat([xx.view(-1, 1) for xx in x], dim=0) for x in grad_list]
            # tmp: Copia temporal dos parámetros do modelo actual
            tmp = [param.data.clone() for param in net.parameters()]
            # weight: Lista de tensores cos parámetros aplanados do modelo actual
            weight = torch.cat([x.view(-1, 1) for x in tmp], dim=0)

            # CALCULAR HESSIAN VECTOR PRODUCT CON LBFGS (A PARTIR DA EPOCA 50)
            if e >= args.det_start:
                hvp = lbfgs_fed_rec(weight_record, grad_record, weight - last_weight)
            else:
                hvp = None

            # ATACAR
            param_list = byz(param_list, undetected_byz_index)

            # SELECCIONAR MÉTODO DE AGREGACIÓN
            grad, distance = select_aggregation(args.aggregation, old_grad_list, param_list, net, lr,
                                                undetected_byz_index, hvp)

            # ACTUALIZAR A DISTANCIA MALICIOSA
            if distance is not None and e > 50:
                malicious_score.append(distance)

            # DETECCION DE CLIENTES MALICIOSOS
            if len(malicious_score) > 10 and args.tipo_exec != 'no_detect':
                mal_scores = np.sum(malicious_score[-10:], axis=0)
                det = detectarMaliciosos(mal_scores, args, para_string, e, total_clients, undetected_byz_index, path)
                if det is not None:
                    datos_finais(path, train_acc_list, test_data_loader, net, device, e, malicious_score,
                                 args.byz_type)
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

            mix_data_labels(each_worker_data, each_worker_label, undetected_byz_index, ben_workers)

            #############################################################################
            # PRECISIÓNS
            # CALCULAR A PRECISIÓN DO ENTRENO CADA 10 ITERACIÓNS
            if (e + 1) % 10 == 0:
                testear_precisions(test_data_loader, net, device, e, train_acc_list, path, target_backdoor_dba,
                                   test_edge_images, args.byz_type)

            # GARDAR AS PUNTUACIÓNS DO ENTRENO CADA 10 ITERACIÓNS
            if (e + 1) % 10 == 0 and e > args.det_start:
                gardar_puntuacions(malicious_score, path)

    # CALCUAR A PRECISIÓN FINAL DO TESTEO
    resumo_final(test_data_loader, net, device, e, malicious_score, path)

    return None
