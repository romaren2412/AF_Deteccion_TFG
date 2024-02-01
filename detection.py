import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

from debuxar_score import *


def detection(score, args, para_string, iter, clients, byz_undetected_index, path_file):
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
        return []

    print('Ataque confirmado!')

    # Imprimir resultados en fichero de texto
    with open(path_file + '/Ataques_Detectados.txt', 'a+') as f:
        f.write("\n------\n" + para_string + "\n")
        f.write("Stop at iteration: " + str(iter) + "\n")
        f.write("acc %0.4f; recall %0.4f; fpr %0.4f; fnr %0.4f;\n" % (acc, recall, fpr, fnr))
        f.write("detected_clients: " + str(detected_clients) + "\n")
        f.write("silhouette: " + str(silhouette) + "\n")

    return detected_clients


def detection2(score, B, K):
    """
    Detecta ataques mediante la Gap Statistics.
    :param score: Suma das últimas 10 puntuacións maliciosas de cada un dos 100 clientes (vector de 100)
    :param B: Número de puntos de referencia para calcular la Gap Statistics
    :param K: Número de clusters posibles
    :return:
    """
    select_k = -1
    ks = range(1, K)  # número de clusters posibles
    gaps = np.zeros(len(ks))  # vector de gaps
    gapDiff = np.zeros(len(ks) - 1)  # vector de diferencias de gaps
    sdk = np.zeros(len(ks))  # vector de desviacións estándar de los gaps

    # Normalizar puntuacións (paper: apply linear transformation on st)
    min = np.min(score)
    max = np.max(score)
    score = (score - min) / (max - min)

    # Bucle para diferentes números de clusters Obxectivo: Se os clústeres con datos reais son mellores que os
    # aleatorios, asúmese que hai unha estrutura. La Gap Statistics (gaps[i]) se calcula como la diferencia entre el
    # logaritmo de la media de las métricas en datos aleatorios y el logaritmo de la métrica en datos reales. Se
    # espera que la Gap Statistics sea alta cuando el número de clústeres (k) sea apropiado para describir la
    # estructura en los datos reales.
    for i, k in enumerate(ks):
        # Aplica KMeans para cada número de clusters e calcula Wk
        estimator = KMeans(n_clusters=k, n_init=10)
        estimator.fit(score.reshape(-1, 1))
        label_pred = estimator.labels_
        center = estimator.cluster_centers_
        Wk = np.sum([np.square(score[m] - center[label_pred[m]]) for m in range(len(score))])

        WkRef = np.zeros(B)  # Inicializacion de WkRef
        for j in range(B):
            # Aplica KMeans a datos aleatorios y calcula WkRef
            rand = np.random.uniform(0, 1, len(score))
            estimator = KMeans(n_clusters=k, n_init=10)
            estimator.fit(rand.reshape(-1, 1))
            label_pred = estimator.labels_
            center = estimator.cluster_centers_
            WkRef[j] = np.sum([np.square(rand[m] - center[label_pred[m]]) for m in range(len(rand))])

        # Calcula la GapStatistics y la desviación estándar
        gaps[i] = np.log(np.mean(WkRef)) - np.log(Wk)
        sdk[i] = np.sqrt((1.0 + B) / B) * np.std(np.log(WkRef))

        # GapDiff: Diferencia entre o gap anterior e o actual xunto coa desviación estándar
        if i > 0:
            gapDiff[i - 1] = gaps[i - 1] - gaps[i] + sdk[i]

    # Busca o número de clusters óptimo (o primeiro punto onde a diferenza entre as métricas é non negativa)
    # Se non hai ataque, será o primeiro (a gapdiff será positiva)
    for i in range(len(gapDiff)):
        if gapDiff[i] >= 0:
            select_k = i + 1
            break

    # Se hai un cluster, non hai ataque
    if select_k == 1:
        print('No attack detected!')
        return 0
    else:
        print('Attack Detected!')
        return 1


def detection1(score):
    """
    Detecta ataques mediante la Gap Statistics.
    :param score: Suma das últimas 10 puntuacións maliciosas de cada un dos 100 clientes (vector de 100)
    :return:
    """
    nrefs = 10  # número de clientes de referencia "score"
    ks = range(1, 8)  # número de clusters posibles
    gaps = np.zeros(len(ks))  # vector de gaps
    gapDiff = np.zeros(len(ks) - 1)  # vector de diferencias de gaps
    sdk = np.zeros(len(ks))  # vector de desviacións estándar de los gaps

    # Normalizar puntuacións (paper: apply linear transformation on st)
    min = np.min(score)
    max = np.max(score)
    score = (score - min) / (max - min)

    select_k = 1

    # Bucle para diferentes números de clusters
    # Obxectivo: Se os clústeres con datos reais son mellores que os aleatorios, asúmese que hai unha estrutura.
    # La Gap Statistics (gaps[i]) se calcula como la diferencia entre el logaritmo de la media de las métricas en datos aleatorios y el logaritmo de la métrica en datos reales.
    # Se espera que la Gap Statistics sea alta cuando el número de clústeres (k) sea apropiado para describir la estructura en los datos reales.
    for i, k in enumerate(ks):
        # Aplica KMeans para cada número de clusters e calcula Wk
        estimator = KMeans(n_clusters=k, n_init=10)
        estimator.fit(score.reshape(-1, 1))
        label_pred = estimator.labels_
        center = estimator.cluster_centers_
        Wk = np.sum([np.square(score[m] - center[label_pred[m]]) for m in range(len(score))])

        WkRef = np.zeros(nrefs)  # Inicializacion de WkRef
        for j in range(nrefs):
            # Aplica KMeans a datos aleatorios y calcula WkRef
            rand = np.random.uniform(0, 1, len(score))
            estimator = KMeans(n_clusters=k, n_init=10)
            estimator.fit(rand.reshape(-1, 1))
            label_pred = estimator.labels_
            center = estimator.cluster_centers_
            WkRef[j] = np.sum([np.square(rand[m] - center[label_pred[m]]) for m in range(len(rand))])

        # Calcula la GapStatistics y la desviación estándar
        gaps[i] = np.log(np.mean(WkRef)) - np.log(Wk)
        sdk[i] = np.sqrt((1.0 + nrefs) / nrefs) * np.std(np.log(WkRef))

        # GapDiff: Diferencia entre o gap anterior e o actual xunto coa desviación estándar
        if i > 0:
            gapDiff[i - 1] = gaps[i - 1] - gaps[i] + sdk[i]

    # Busca o número de clusters óptimo (o primeiro punto onde a diferenza entre as métricas é non negativa)
    # Se non hai ataque, será o primeiro (a gapdiff será positiva)
    for i in range(len(gapDiff)):
        if gapDiff[i] >= 0:
            select_k = i + 1
            break

    # Se hai un cluster, non hai ataque
    if select_k == 1:
        print('No attack detected!')
        return 0
    else:
        print('Attack Detected!')
        return 1


def detectarMaliciosos1(mal_scores, args, para_string, e, total_clients, undetected_byz_index, path_file, graf):
    # Detección orixinal, usando 'detection1'
    if graf:
        path_Grafica = path_file + '/GardarGraficas/'
        if not os.path.exists(path_Grafica):
            os.makedirs(path_Grafica)

    if detection1(mal_scores):
        print('Stop at iteration:', e)
        det = detection(mal_scores, args, para_string, e, total_clients, undetected_byz_index, path_file)
        # Métrica de detección (silhouette)
        if graf:
            grafica_gardar(mal_scores, e, args.byz_type, path_Grafica)
        if len(det) > 0:
            return det

    return None

def detectarMaliciosos2(mal_scores, args, para_string, e, total_clients, undetected_byz_index, path_file, graf):
    # Detección modificada, usando 'detection2'
    if graf:
        path_grafica = path_file + '/GardarGraficas/'
        if not os.path.exists(path_grafica):
            os.makedirs(path_grafica)

    if detection2(mal_scores, B=20, K=10):
        print('Stop at iteration:', e)
        det = detection(mal_scores, args, para_string, e, total_clients, undetected_byz_index, path_file)
        # Métrica de detección (silhouette)
        if graf:
            grafica_gardar(mal_scores, e, args.byz_type, path_grafica)
        if len(det) > 0:
            return det
    return None

def detectarMaliciososComp(mal_scores, args, para_string, e, total_clients, undetected_byz_index, path_file):
    # Imprimir resultados en fichero de texto
    if not os.path.exists(path_file):
        os.makedirs(path_file)

    det1 = detection1(mal_scores)
    det2 = detection2(mal_scores, B=20, K=10)

    # Imprimir resultados en fichero de texto
    with open(path_file + 'Det_Compar.txt', 'a+') as f:
        f.write("\n------\n")
        f.write(f"Na iteración {e}: det1 = {det1}, det2 = {det2}\n")

    if det1 or det2:
        print('Stop at iteration:', e)
        grafica_gardar(mal_scores, e, args.byz_type, path_file)
        det = detection(mal_scores, args, para_string, e, total_clients, undetected_byz_index, path_file)
        # Métrica de detección (silhouette)
        if len(det) > 0:
            with open(path_file + 'Det_Compar.txt', 'a+') as f:
                f.write(f"O ataque foi detectado con seguridade\n")
            return det
        else:
            with open(path_file + 'Det_Compar.txt', 'a+') as f:
                f.write(f"O ataque foi detectado pero con pouca seguridade\n")


def detectionAux(score, args, para_string, iter, clients, byz_undetected_index, path_file):
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
        return []

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
        path_Grafica = path_file + '/GardarGraficas/'
        if not os.path.exists(path_Grafica):
            os.makedirs(path_Grafica)

    if detection1(mal_scores):
        print('Stop at iteration:', e)
        return detectionAux(mal_scores, args, para_string, e, total_clients, undetected_byz_index, path_file)


def detectarMaliciosos(m, a, p, e, c, b, path_file, tipo, graf):
    if tipo == 1:
        return detectarMaliciosos1(m, a, p, e, c, b, path_file, graf)
    elif tipo == 2:
        return detectarMaliciosos2(m, a, p, e, c, b, path_file, graf)
    elif tipo == 3:
        return detectarMaliciososComp(m, a, p, e, c, b, path_file)
    elif tipo == 4:
        return detectarMaliciososOriginal(m, a, p, e, c, b, path_file, graf)
    else:
        raise Exception("Tipo de detección non válido")
