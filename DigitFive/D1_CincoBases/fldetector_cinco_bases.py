import datetime

import torch.nn as nn
from clases_redes import DigitFiveNet
from detection import *
from DigitFive.digit_five import *
import copy
from np_aggregation import *
from DigitFive.arquivos import *
from lbfgs import lbfgs
from MNIST.byzantine import select_byzantine_range


def show_batch(data_loader, show_function, num_images=3):
    # Obtener un lote de datos del DataLoader
    data_iter = iter(data_loader)
    images, labels = next(data_iter)

    # Mostrar las primeras 'num_images' imágenes con sus etiquetas
    for i in range(num_images):
        img = images[i].cpu().numpy()
        label = labels[i].item()
        show_function(img, label)


def mostrar_imaxes(aprendedores, undetected_byz_index, ben_workers):
    byz_rand = np.random.choice(undetected_byz_index, 1)[0]
    ben_rand = np.random.choice(ben_workers, 1)[0]
    show_batch(aprendedores[byz_rand].trainloader, show, num_images=8)
    show_batch(aprendedores[ben_rand].trainloader, show, num_images=5)


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
    para_string = "nepochs: " + str(c.EPOCH) + ", lr: " + str(c.LR) + ", nworkers: " + str(c.SIZE) + ", nbyz: " + str(
        len(byz_workers))

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join('PROBAS/D5/5Dif', args.home_path, timestamp, args.aggregation, str(entrenamento))
    if not os.path.exists(path):
        os.makedirs(path)
    precision_path = path + '/Precision.txt'
    ataques_path = path + '/Ataques_Detectados.txt'
    with (open(precision_path, 'w+')) as f:
        f.write(para_string + '\n')

    # EJECUCIÓN
    with device:
        # ARQUITECTURA DO MODELO - CNN
        global_net = DigitFiveNet()
        global_net.initialize_weights()

        # Mueve el modelo a la GPU o CPU según el contexto
        global_net.to(device)

        ########################################################################################################
        # CARGA DO DATASET
        aprendedores = []
        for i in range(len(total_clients)):
            c_ = copy.deepcopy(c)
            c_.RANK = i
            c_.recalculate_db_5bases()
            print(f"[INFO USER {i}] Loading data...")
            ap = Digit_Five_training(c_)
            ap.create_train()
            ap.create_test()
            if i in byz_workers:
                ap.byz = True
            aprendedores.append(ap)

        testloader_global = aprendedores[0].create_test_global()

        ####################################################################################################

        # DECIDIR EL TIPO DE ATAQUE
        byz = select_byzantine_range(args.byz_type)

        # Definir la pérdida de entropía cruzada softmax
        softmax_cross_entropy = nn.CrossEntropyLoss()

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

        epochs = c.EPOCH
        lr = c.LR
        grad_list = []
        old_grad_list = []
        weight_record = []
        grad_record = []
        malicious_score = []
        precision_array = []
        tipo_bases = 'cinco_diferentes'

        ###################################################################################################

        # EXECUTAR ATAQUES
        target_backdoor_dba = 0
        if args.byz_type == 'dba':
            for index, g in enumerate(np.array_split(undetected_byz, 4)):
                for byzantine in g:
                    aprendedores[byzantine].dba_index = index

        # MOSTRAR IMAXES
        # mostrar_imaxes(aprendedores, undetected_byz_index, ben_workers)

        # ##################################################################################################################
        print("COMEZO DO ADESTRAMENTO...")
        # CADA ÉPOCA
        for e in range(epochs):
            print("EPOCH: ", e)
            # CADA CLIENTE
            for ap in aprendedores:
                ap.sl.adestrar(softmax_cross_entropy, global_net, grad_list, args.byz_type, target_backdoor_dba)

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

            # ATACAR
            param_list = byz(param_list, undetected_byz_index)

            # SELECCIONAR MÉTODO DE AGREGACIÓN
            grad, distance = select_aggregation(args.aggregation, old_grad_list, param_list, global_net, lr,
                                                undetected_byz_index, hvp)

            # ACTUALIZAR A DISTANCIA MALICIOSA
            if distance is not None and e > c.FLDET_START:
                malicious_score.append(distance)

            # DETECCION DE CLIENTES MALICIOSOS
            if len(malicious_score) > 10 and args.tipo_exec != 'no_detect':
                mal_scores = np.sum(malicious_score[-10:], axis=0)
                det = detectar_maliciosos(mal_scores, args, para_string, e, total_clients, undetected_byz_index, path)
                if det is not None:
                    datos_finais(path, precision_array, testloader_global, global_net, device, e, malicious_score,
                                 args.byz_type)
                    return det

            # MOSTRAR MAL_SCORES
            # grafica_cluster(malicious_score, undetected_byz_index, e)

            # lr = actualizar_lr(e, lr)

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
            if (e + 1) % 200 == 0:
                testear_precisions(aprendedores, testloader_global, global_net, device, e, precision_array, path,
                                   args.data_type, args.extra_data_type, tipo_bases)

            # GARDAR AS PUNTUACIÓNS DO ENTRENO CADA 10 ITERACIÓNS
            # if (e + 1) % 10 == 0 and e > c.FLDET_START:
            # gardar_puntuacions(malicious_score, path)

        resumo_final(testloader_global, global_net, device, e, malicious_score, path)
    return None
