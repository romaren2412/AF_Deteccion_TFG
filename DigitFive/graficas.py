# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np

import Miguel.config
from Miguel.config import parse_args

args = parse_args()
FLDET_START = Miguel.config.Config().FLDET_START


def debuxar_medias(opt, data, undetected_byz_index):
    epochs = list(range(1, len(data[0]) + 1))
    epochs = [tick + FLDET_START for tick in epochs]
    benign_data = [client_data for i, client_data in enumerate(data) if i not in undetected_byz_index]
    mean_benign = np.mean(benign_data, axis=0)
    mean_byz = np.mean([client_data for i, client_data in enumerate(data) if i in undetected_byz_index], axis=0)

    if opt["tipo"] == "catro_bases":
        dbe = opt["data_ben"]
        dbz = opt["data_byz"]
        plt.scatter(epochs, mean_benign, label=f'{dbe} (media)', color='green', marker='o')
        plt.scatter(epochs, mean_byz, label=f'{dbz}', color='red', marker='x')
        plt.title(f'Evolución das Malicious Scores - {dbe}/{dbz}')
    else:
        at = opt["ataque"]
        plt.scatter(epochs, mean_benign, label='Media dos benignos', color='green', marker='o')
        plt.scatter(epochs, mean_byz, label='Media dos bizantinos', color='red', marker='x')
        plt.title(f'Evolución das Malicious Scores - {at}')

    if opt["detectar"]:
        plt.axvline(x=11 + FLDET_START, color='gray', linestyle='dotted', linewidth=1.5, label='Comezo de deteccións')

    plt.xlabel('Iteración')
    plt.ylabel('Malicious Score')

    plt.legend(loc='upper left')

    if opt["gardar"]:
        plt.savefig(opt["save_path"] + '/medias.png')
    plt.show()


def grafica_cluster(opt, mal_scores, undetected_byz_index):
    acum_scores = [np.sum(client_scores[-10:]) for client_scores in mal_scores]
    colors = ['red' if i in undetected_byz_index else 'blue' for i in range(len(acum_scores))]
    plt.scatter(range(1, len(mal_scores) + 1), acum_scores, c=colors, marker='o')

    # Set the title and labels
    plt.title('Puntuacións maliciosas acumuladas para a clusterización- {}'.format(opt["ataque"]))
    plt.xlabel('Cliente')
    plt.ylabel('Malicious Score')

    # Create custom legend
    clientes_bizantinos = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10)
    clientes_benignos = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10)
    plt.legend([clientes_bizantinos, clientes_benignos], ['Clientes bizantinos', 'Clientes benignos'])

    if opt["gardar"]:
        plt.savefig(opt["save_path"] + '/cluster.png')
    plt.show()


def grafica_maliciosas_epoca(opt, mal_scores, undetected_byz_index, epoch=-1):
    if epoch != -1:
        epoch = epoch - FLDET_START

    # Colorear según si es byzantino o no
    colores = ['red' if i in undetected_byz_index else 'blue' for i in range(len(mal_scores))]
    last_iter = [client[epoch] for client in mal_scores]
    plt.scatter(range(len(mal_scores)), last_iter, color=colores)

    # Añadir leyenda manualmente
    clientes_bizantinos = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10)
    clientes_benignos = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10)
    plt.legend([clientes_bizantinos, clientes_benignos], ['Clientes byzantinos', 'Clientes benignos'])

    # Dibujar el ndarray en un gráfico
    plt.xlabel('Índice do cliente')
    plt.ylabel('Malicious score')
    if epoch == -1:
        plt.title('Puntuacións maliciosas na iteración da clusterización')
    else:
        plt.title(f'Puntuacións maliciosas na iteración {epoch}')

    if opt["gardar"]:
        plt.savefig(opt["save_path"] + '/cluster.png')
    plt.show()


def debuxar_precision_catro_bases(opt, datos, data_ben, data_byz):
    datos.columns = ["Iteracion", "ACC_BEN", "ACC_Extra", "ACC_Global"]

    plt.figure(figsize=(10, 6))
    plt.plot(datos['Iteracion'], datos['ACC_BEN'], marker='o', label=f'{data_ben} [4]', linestyle='dashed', color='blue')
    plt.plot(datos['Iteracion'], datos['ACC_Extra'], marker='o', label=f'{data_byz} [1]', linestyle='dashed', color='red')

    plt.title(f'Evolución da precisión - {data_ben}/{data_byz}')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid(True)

    if opt["gardar"]:
        plt.savefig(opt["save_path"] + '/precisions.png')
    plt.show()


def debuxar_precision_ataque(opt, datos):
    attack_type = opt["ataque"]
    # Asignar nombres a las columnas
    if attack_type in ('backdoor', 'dba', 'backdoor_sen_pixel', 'edge'):
        targeted = True
        datos.columns = ["Iteracion", "ACC_Global", "ASR"]
    else:
        targeted = False
        datos.columns = ["Iteracion", "ACC_Global"]

    # Crear la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(datos['Iteracion'], datos['ACC_Global'], marker='^', label='ACC Global')
    if targeted:
        plt.plot(datos['Iteracion'], datos['ASR'], marker='o', color='red', label='Éxito do ataque', linestyle='dashed')

    # Configurar la gráfica
    plt.title(f'Evolución da precisión - {attack_type}')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid(True)

    if opt["gardar"]:
        plt.savefig(opt["save_path"] + '/precisions.png')
    plt.show()


def grafica_cluster_ataque(opt, mal_scores, undetected_byz_index):
    acum_scores = [np.sum(client_scores[-10:]) for client_scores in mal_scores]
    colors = ['red' if i in undetected_byz_index else 'blue' for i in range(len(acum_scores))]
    plt.scatter(range(1, len(mal_scores) + 1), acum_scores, c=colors, marker='o')

    # Set the title and labels
    plt.title(f'Puntuacións maliciosas acumuladas para a clusterización - {opt["ataque"]}')
    plt.xlabel('Cliente')
    plt.ylabel('Malicious Score')

    # Create custom legend
    clientes_bizantinos = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10)
    clientes_benignos = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10)
    plt.legend([clientes_bizantinos, clientes_benignos], ['Clientes bizantinos', 'Clientes benignos'])

    if opt["gardar"]:
        plt.savefig(opt["save_path"] + '/cluster.png')
    plt.show()


def leer_undetected(archivo):
    with open(archivo, 'r') as file:
        for linea in file:
            if "Byzantinos (non detectados aínda):" in linea:
                byzantinos_str = linea.split(":")[1].strip()
                if byzantinos_str == "[]":
                    return None
                else:
                    return [int(x) for x in byzantinos_str.strip("[]").split(",")]
