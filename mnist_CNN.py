import csv
import torch.nn as nn
import torch.nn.init as init
import torchvision
from torchvision import transforms
import datetime

import np_aggregation
import clases_redes as cr

from lbfgs import *
from byzantine import *
from detection import *
from evaluate import *


######################################################################
# SELECCIÓNS DE ATAQUES E AGREGACIÓNS
def select_byzantine_range(t):
    # decide attack type
    if t == 'partial_trim':
        # partial knowledge trim attack
        return partial_trim_range
    elif t == 'full_trim':
        # full knowledge trim attack
        return full_trim_range
    elif t == 'no':
        return no_byz_range
    elif t == 'gaussian':
        return gaussian_attack_range
    elif t == 'mean_attack':
        return mean_attack_range
    elif t == 'full_mean_attack':
        return full_mean_attack_range
    elif t == 'dir_partial_krum_lambda':
        return dir_partial_krum_lambda_range
    elif t == 'dir_full_krum_lambda':
        return dir_full_krum_lambda_range
    elif t in ('backdoor', 'dba', 'edge'):
        return scaling_attack_range
    elif t == 'label_flip':
        return no_byz_range
    else:
        raise NotImplementedError


def select_aggregation(t, old_gradients, param_list, net, lr, b, hvp=None):
    # decide aggregation type
    if t == 'simple_mean':
        return np_aggregation.simple_mean(old_gradients, param_list, net, lr, b, hvp)
    elif t == 'trim':
        return np_aggregation.trim(old_gradients, param_list, net, lr, b, hvp)
    elif t == 'krum':
        return np_aggregation.krum(old_gradients, param_list, net, lr, b, hvp)
    elif t == 'median':
        return np_aggregation.median(old_gradients, param_list, net, lr, b, hvp)
    else:
        raise NotImplementedError


######################################################################
# DISTRIBUCIÓN DE DATOS
def repartir_datos(args, train_data_loader, num_workers, device):
    # ASIGNACIÓN ALEATORIA DOS DATOS ENTRE OS CLIENTES
    # Semilla
    seed = args.seed
    np.random.seed(seed)

    bias_weight = args.bias
    other_group_size = (1 - bias_weight) / 9.
    worker_per_group = num_workers / 10

    each_worker_data = [[] for _ in range(num_workers)]
    each_worker_label = [[] for _ in range(num_workers)]

    for _, (data, label) in enumerate(train_data_loader):
        data, label = data.to(device), label.to(device)
        for (x, y) in zip(data, label):
            x = x.to(device).view(1, 1, 28, 28)
            y = y.to(device).view(-1)

            # Asignar un punto de datos a un grupo
            upper_bound = (y.cpu().numpy()) * other_group_size + bias_weight
            lower_bound = (y.cpu().numpy()) * other_group_size
            rd = np.random.random_sample()

            if rd > upper_bound:
                worker_group = int(np.floor((rd - upper_bound) / other_group_size) + y.cpu().numpy() + 1)
            elif rd < lower_bound:
                worker_group = int(np.floor(rd / other_group_size))
            else:
                worker_group = y.cpu().numpy()

            # Asignar un punto de datos a un trabajador
            rd = np.random.random_sample()
            selected_worker = int(worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
            each_worker_data[selected_worker].append(x)
            each_worker_label[selected_worker].append(y)

    # Concatenar los datos para cada trabajador para evitar huecos
    each_worker_data = [torch.cat(each_worker, dim=0) for each_worker in each_worker_data]
    each_worker_label = [torch.cat(each_worker, dim=0) for each_worker in each_worker_label]

    # Barajar aleatoriamente los trabajadores
    random_order = np.random.RandomState(seed=seed).permutation(num_workers)
    each_worker_data = [each_worker_data[i] for i in random_order]
    each_worker_label = [each_worker_label[i] for i in random_order]

    return each_worker_data, each_worker_label


######################################################################
# MOSTRAR PRECISIÓNS + GARDAR DATOS
def gardar_precision(precision_path, train_acc_list):
    with (open(precision_path, 'a')) as f:
        np.savetxt(f, train_acc_list, fmt='%.4f')


def resumo_final(test_data_loader, net, device, e, malicious_score, path):
    test_accuracy = evaluate_accuracy(test_data_loader, net, device)
    print("Epoch %02d. Test_acc %0.4f" % (e, test_accuracy))
    with open(path + '/score1.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(malicious_score)


def datos_finais(precision_path, train_acc_list, test_data_loader, net, device, e, malicious_score, file_path):
    # Precisión entreno
    gardar_precision(precision_path, train_acc_list)
    # PRECISION FINAL + Malicious score
    resumo_final(test_data_loader, net, device, e, malicious_score, file_path)



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
        args.lr) + ", batch_size: " + str(args.batch_size) + ", nworkers: " + str(
        args.nworkers) + ", nbyz: " + str(args.nbyz)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join('PROBAS', timestamp, args.home_path, args.byz_type, args.aggregation, str(entrenamento))
    if not os.path.exists(path):
        os.makedirs(path)
    precision_path = path + '/Precision.txt'
    ataques_path = path + '/Ataques_Detectados.txt'
    with (open(precision_path, 'w+')) as f:
        f.write(para_string + '\n')

    # EJECUCIÓN
    with device:
        num_outputs = 10

        # ARQUITECTURA DO MODELO - CNN
        net = cr.CNN(num_channels=1, num_outputs=num_outputs)

        # INICIALIZAR PESOS
        def init_weights(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                init.xavier_uniform_(m.weight.data, gain=2.24)
                init.constant_(m.bias.data, 0.0)

        net.apply(init_weights)

        # Mueve el modelo a la GPU o CPU según el contexto
        net.to(device)

        ########################################################################################################
        # CARGA DO DATASET
        # Definir la transformación para normalizar los datos
        transform = transforms.Compose([transforms.ToTensor()])
        # Cargar el conjunto de datos de entrenamiento
        train_data = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=60000, shuffle=True,
                                                        generator=torch.Generator(device='cuda'))
        # Cargar el conjunto de datos de prueba
        test_data = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=500, shuffle=False)

        if args.byz_type == 'edge':
            digit_edge = 7
            indices_digit = torch.nonzero(test_data.targets == digit_edge, as_tuple=False)[:, 0]
            images_digit = test_data.data[indices_digit, :] / 255.0
            test_edge_images = images_digit.view(-1, 1, 28, 28).to(device)
            label = torch.ones(len(test_edge_images)).to(torch.int).to(device)

        ####################################################################################################

        # DECIDIR EL TIPO DE ATAQUE
        byz = select_byzantine_range(args.byz_type)

        # Definir la pérdida de entropía cruzada softmax
        softmax_cross_entropy = nn.CrossEntropyLoss()

        # set upt parameters
        num_workers = len(total_clients)
        byz_workers = [i for i in range(10, args.nbyz + 10)]
        # byz_workers = [i for i in range(100) if i % 4 == 0]
        # byz_workers = [i for i in range(args.nbyz)]
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

        ###################################################################################################

        # ASIGNACIÓN ALEATORIA DOS DATOS ENTRE OS CLIENTES
        each_worker_data, each_worker_label = repartir_datos(args, train_data_loader, num_workers, device)

        ######################################################################################################

        # EXECUTAR ATAQUES
        target_backdoor_dba = 0
        if args.byz_type == 'label_flip':
            each_worker_label = label_flip_range(each_worker_label, undetected_byz_index)
        elif args.byz_type == 'backdoor':
            each_worker_data, each_worker_label = backdoor_range(each_worker_data, each_worker_label,
                                                                 undetected_byz_index, target_backdoor_dba)
        elif args.byz_type == 'edge':
            each_worker_data, each_worker_label = edge_range(each_worker_data, each_worker_label,
                                                             undetected_byz_index, test_edge_images,
                                                             label)
        elif args.byz_type == 'dba':
            each_worker_data, each_worker_label = dba_range(each_worker_data, each_worker_label,
                                                            undetected_byz_index, target_backdoor_dba)

        # ##################################################################################################################
        # set malicious scores
        malicious_score = []

        # CADA ÉPOCA
        for e in range(epochs):
            # CADA CLIENTE
            for i in range(num_workers):
                inputs, labels = each_worker_data[i][:], each_worker_label[i][:]
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
            if e > 50:
                hvp = lbfgs_fed_rec(weight_record, grad_record, weight - last_weight)
            else:
                hvp = None

            # ATACAR
            if e % 100 >= 50:
                param_list = byz(param_list, undetected_byz_index)

            # SELECCIONAR MÉTODO DE AGREGACIÓN
            grad, distance = select_aggregation(args.aggregation, old_grad_list, param_list, net, lr,
                                                undetected_byz_index, hvp)

            # ACTUALIZAR A DISTANCIA MALICIOSA
            if distance is not None and e > 50:
                malicious_score.append(distance)

            # DETECCION DE CLIENTES MALICIOSOS
            if len(malicious_score) > 10:
                mal_scores = np.sum(malicious_score[-10:], axis=0)
                det = detectarMaliciosos(mal_scores, args, para_string, e, total_clients, undetected_byz_index, path)
                if det is not None:
                    datos_finais(precision_path, train_acc_list, test_data_loader, net, device, e, malicious_score,
                                 path)
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
            if (e + 1) % 10 == 0:
                train_accuracy = evaluate_accuracy(test_data_loader, net, device)
                if args.byz_type == ('backdoor' or 'dba'):
                    backdoor_sr = evaluate_backdoor(test_data_loader, net, target=target_backdoor_dba, device=device)
                    print("Epoch %02d. Train_acc %0.4f Attack_sr %0.4f" % (e, train_accuracy, backdoor_sr))
                elif args.byz_type == 'edge':
                    backdoor_sr = evaluate_edge_backdoor(test_edge_images, net, device)
                    print("Epoch %02d. Train_acc %0.4f Attack_sr %0.4f" % (e, train_accuracy, backdoor_sr))
                else:
                    print("Epoch %02d. Train_acc %0.4f" % (e, train_accuracy))
                train_acc_list.append(train_accuracy)

            # GARDAR A PRECISIÓN DO ENTRENO CADA 50 ITERACIÓNS
            if (e + 1) % 50 == 0:
                gardar_precision(precision_path, train_acc_list)

            # CALCUAR A PRECISIÓN FINAL DO TESTEO
            if (e + 1) == args.nepochs:
                resumo_final(test_data_loader, net, device, e, malicious_score, path)

    return None
