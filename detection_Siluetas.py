from detection import *


def detectionAux(score, para_string, it, clients, byz_undetected_index, path_file):
    """
    Detecta ataques mediante clustering.
    :param score: vector da suma das últimas 10 puntuacións maliciosas
    :param para_string: string para crear ficheiro de texto
    :param it: iteración na que se detecta o ataque
    :param clients: lista de clientes
    :param byz_undetected_index: índices dos clientes byzantinos non detectados
    """

    nbyz = len(byz_undetected_index)
    clients_temp = [i for i in range(len(clients))]

    # Normalizar puntuacións (paper: apply linear transformation on st)
    min = np.min(score)
    max = np.max(score)
    score = (score - min) / (max - min)

    # Estimar el número de clusters (2: benigno y malicioso)
    estimator = KMeans(n_clusters=2, n_init=10)
    estimator.fit(score.reshape(-1, 1))
    label_pred = estimator.labels_

    # Puntuacións dos clusters (0: malicioso, 1: benigno)
    # Se hai máis acertos no cluster 0 que no 1, intercambiar os clusters
    if np.mean(score[label_pred == 0]) < np.mean(score[label_pred == 1]):
        label_pred = 1 - label_pred
    real_label = np.ones(len(clients))
    real_label[byz_undetected_index] = 0

    # Precisión: traballadores totais ben clasificados
    acc = len(label_pred[label_pred == real_label]) / len(clients)

    # Recall: traballadores byzantinos ben identificados
    # FNR: Tasa de Falsos Negativos
    # 2 casos posibles:
    # 1. Se nbyz = 0, detecta un ataque falso (non hai byzantinos, nin ben nin mal identificados)
    # 2. Se nbyz > 0, detecta un ataque que pode ser real
    if nbyz == 0:
        recall = -1
        fnr = -1
    else:
        recall = 1 - np.sum(label_pred[byz_undetected_index]) / nbyz
        fnr = np.sum(label_pred[byz_undetected_index]) / nbyz

    # FPR: Tasa de Falsos Positivos
    fpr = 1 - np.sum(label_pred[clients_temp]) / (len(clients) - nbyz)

    print("acc %0.4f; recall %0.4f; fpr %0.4f; fnr %0.4f;" % (acc, recall, fpr, fnr))

    # Silhouette score: canto máis preto do 1, mellor calidade da agrupación en 2 clusters
    silhouette = silhouette_score(score.reshape(-1, 1), label_pred)
    print(silhouette)

    # Imprimir clientes maliciosos
    index_detected_clients = np.where(label_pred == 0)[0]
    detected_clients = [clients[i] for i in index_detected_clients]
    print("Clientes detectados en cluster malicioso: ", detected_clients)

    # Imprimir resultados en fichero de texto
    if not os.path.exists(path_file):
        os.makedirs(path_file)

    print('Ataque confirmado!')

    # Imprimir resultados en fichero de texto
    with open(path_file + '/Ataques_Detectados.txt', 'a+') as f:
        f.write("\n------\n" + para_string + "\n")
        f.write("Stop at iteration: " + str(it) + "\n")
        f.write("acc %0.4f; recall %0.4f; fpr %0.4f; fnr %0.4f;\n" % (acc, recall, fpr, fnr))
        f.write("detected_clients: " + str(detected_clients) + "\n")
        f.write("silhouette: " + str(silhouette) + "\n")

    return detected_clients


def detectionAux2(score, args, para_string, iter, clients, byz_undetected_index, path_file):
    """
    Detecta ataques mediante clustering.
    :param score: vector da suma das últimas 10 puntuacións maliciosas
    :param args: argumentos de entrada
    :param para_string: string para crear ficheiro de texto
    :param iter: iteración na que se detecta o ataque
    :param clients: lista de clientes
    :param byz_undetected_index: índices dos clientes byzantinos non detectados
    """

    nbyz = len(byz_undetected_index)
    clients_temp = [i for i in range(len(clients))]

    # Normalizar puntuacións (paper: apply linear transformation on st)
    min = np.min(score)
    max = np.max(score)
    score = (score - min) / (max - min)

    # Estimar el número de clusters (2: benigno y malicioso)
    estimator = KMeans(n_clusters=2, n_init=10)
    estimator.fit(score.reshape(-1, 1))
    label_pred = estimator.labels_

    # Puntuacións dos clusters (0: malicioso, 1: benigno)
    # Se hai máis acertos no cluster 0 que no 1, intercambiar os clusters
    if np.mean(score[label_pred == 0]) < np.mean(score[label_pred == 1]):
        label_pred = 1 - label_pred
    real_label = np.ones(len(clients))
    real_label[byz_undetected_index] = 0

    # Precisión: traballadores totais ben clasificados
    acc = len(label_pred[label_pred == real_label]) / len(clients)

    # Recall: traballadores byzantinos ben identificados
    # FNR: Tasa de Falsos Negativos
    # 2 casos posibles:
    # 1. Se nbyz = 0, detecta un ataque falso (non hai byzantinos, nin ben nin mal identificados)
    # 2. Se nbyz > 0, detecta un ataque que pode ser real
    if nbyz == 0:
        recall = -1
        fnr = -1
    else:
        recall = 1 - np.sum(label_pred[byz_undetected_index]) / nbyz
        fnr = np.sum(label_pred[byz_undetected_index]) / nbyz

    # FPR: Tasa de Falsos Positivos
    fpr = 1 - np.sum(label_pred[clients_temp]) / (len(clients) - nbyz)

    print("acc %0.4f; recall %0.4f; fpr %0.4f; fnr %0.4f;" % (acc, recall, fpr, fnr))

    # Silhouette score: canto máis preto do 1, mellor calidade da agrupación en 2 clusters
    silhouette = silhouette_score(score.reshape(-1, 1), label_pred)
    print(silhouette)

    # Imprimir clientes maliciosos
    index_detected_clients = np.where(label_pred == 0)[0]
    detected_clients = [clients[i] for i in index_detected_clients]
    print("Clientes detectados en cluster malicioso: ", detected_clients)

    # Imprimir resultados en fichero de texto
    if not os.path.exists(path_file):
        os.makedirs(path_file)

    if silhouette < args.silhouette:
        print('Ataque identificado pero con pouca seguridade (silhouette)')
        with open(path_file + '/Ataques_Detectados.txt', 'a+') as f:
            f.write("\n------\n" + para_string + "\n")
            f.write("Ataque identificado pero con pouca seguridade (silhouette) na iteración " + str(iter) + "\n")
            f.write("acc %0.4f; recall %0.4f; fpr %0.4f; fnr %0.4f;\n" % (acc, recall, fpr, fnr))
            f.write("detected_clients: " + str(detected_clients) + "\n")
            f.write("silhouette: " + str(silhouette) + "\n")
        return None

    print('Ataque confirmado!')

    # Imprimir resultados en fichero de texto
    with open(path_file + '/Ataques_Detectados.txt', 'a+') as f:
        f.write("\n------\n" + para_string + "\n")
        f.write("Stop at iteration: " + str(iter) + "\n")
        f.write("acc %0.4f; recall %0.4f; fpr %0.4f; fnr %0.4f;\n" % (acc, recall, fpr, fnr))
        f.write("detected_clients: " + str(detected_clients) + "\n")
        f.write("silhouette: " + str(silhouette) + "\n")

    return detected_clients


def detectarMaliciososOriginal(mal_scores, args, para_string, e, total_clients, undetected_byz_index, path_file, graf):
    # Detección orixinal, usando 'detection1'
    if graf:
        path_grafica = path_file + '/GardarGraficas/'
        if not os.path.exists(path_grafica):
            os.makedirs(path_grafica)

    if detection1(mal_scores):
        print('Stop at iteration:', e)
        det = detectionAux(mal_scores, para_string, e, total_clients, undetected_byz_index, path_file)
        if graf:
            grafica_gardar_byz(mal_scores, e, args.byz_type, path_grafica, undetected_byz_index)
        return det


def detectarMaliciososOriginal2(mal_scores, args, para_string, e, total_clients, undetected_byz_index, path_file, graf):
    # Detección orixinal, usando 'detection1'
    if graf:
        path_grafica = path_file + '/GardarGraficas/'
        if not os.path.exists(path_grafica):
            os.makedirs(path_grafica)

    if detection1(mal_scores):
        print('Stop at iteration:', e)
        det = detectionAux2(mal_scores, args, para_string, e, total_clients, undetected_byz_index, path_file)
        if graf:
            grafica_gardar_byz(mal_scores, e, args.byz_type, path_grafica, undetected_byz_index)
        return det
