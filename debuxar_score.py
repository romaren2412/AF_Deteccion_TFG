import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

FLDET_START = 50


def debuxar_medias(data, undetected_byz_index):
    epochs = list(range(1, len(data[0]) + 1))
    # Ajustar los valores de los ticks sumándoles 50
    epochs = [tick + FLDET_START for tick in epochs]

    # Calcular la media de los clientes benignos
    benign_data = [client_data for i, client_data in enumerate(data) if i not in undetected_byz_index]
    mean_benign = np.mean(benign_data, axis=0)

    # Calcula a media dos clientes bizantinos
    mean_byz = np.mean([client_data for i, client_data in enumerate(data) if i in undetected_byz_index], axis=0)

    # Representar la media de los clientes benignos
    plt.scatter(epochs, mean_benign, label='Media dos benignos', color='green', marker='o')
    plt.scatter(epochs, mean_byz, label='Media dos bizantinos', color='red', marker='x')

    # Añadir una línea constante en la iteración 11
    if detectar:
        plt.axvline(x=11 + FLDET_START, color='gray', linestyle='dotted', linewidth=1.5, label='Comezo de deteccións')

    plt.xlabel('Iteración')
    plt.ylabel('Malicious Score')
    plt.title('Evolución das Malicious Scores - {}'.format(attack_type))

    # Añadir leyenda manualmente
    plt.legend(loc='upper left')

    if gardar:
        plt.savefig(save_path + '/medias.png')
    plt.show()


def grafica_cluster(mal_scores, undetected_byz_index):
    acum_scores = [np.sum(client_scores[-10:]) for client_scores in mal_scores]
    colors = ['red' if i in undetected_byz_index else 'blue' for i in range(len(acum_scores))]
    plt.scatter(range(1, len(mal_scores) + 1), acum_scores, c=colors, marker='o')

    # Set the title and labels
    plt.title('Puntuacións maliciosas acumuladas para a clusterización- {}'.format(attack_type))
    plt.xlabel('Cliente')
    plt.ylabel('Malicious Score')

    # Create custom legend
    clientes_bizantinos = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10)
    clientes_benignos = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10)
    plt.legend([clientes_bizantinos, clientes_benignos], ['Clientes bizantinos', 'Clientes benignos'])

    if gardar:
        plt.savefig(save_path + '/cluster.png')
    plt.show()


def grafica_maliciosas_epoca(mal_scores, undetected_byz_index, epoch=-1):
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

    if gardar:
        plt.savefig(save_path + '/cluster.png')
    plt.show()


def debuxar_precision(datos):
    # Asignar nombres a las columnas
    datos.columns = ["Iteracion", "ACC_Global"]

    # Crear la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(datos['Iteracion'], datos['ACC_Global'], marker='^', label='ACC Global')

    # Configurar la gráfica
    plt.title('Evolución da precisión - {}'.format(attack_type))
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid(True)

    if gardar:
        plt.savefig(save_path + '/precisions.png')
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


if __name__ == "__main__":
    gardar = True
    detectar = False

    path = 'PROBAS/ND/20240324-213911/median/backdoor/0'
    save_path = os.path.dirname(path)
    attack_type = path.split('/')[-2]
    # leer desde la ruta
    data_score = pd.read_csv(path + '/score.csv', header=None).T.values.tolist()
    data_acc = pd.read_csv(path + '/acc.csv', header=0)
    undetected = leer_undetected(path + '/Ataques_Detectados.txt')
    debuxar_medias(data_score, undetected)
    grafica_cluster(data_score, undetected)
    # grafica_maliciosas_epoca(data_score, undetected)
    debuxar_precision(data_acc)
