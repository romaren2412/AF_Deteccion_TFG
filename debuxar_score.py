import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

FLDET_START = 50

def debuxar_medias(data, type, undetected_byz_index):
    iter = list(range(1, len(data[0]) + 1))
    # Ajustar los valores de los ticks sumándoles 50
    iter = [tick + FLDET_START for tick in iter]

    # Calcular la media de los clientes benignos
    benign_data = [client_data for i, client_data in enumerate(data) if i not in undetected_byz_index]
    mean_benign = np.mean(benign_data, axis=0)

    # Calcular la media de los clientes malignos
    byz_data = [client_data for i, client_data in enumerate(data) if i in undetected_byz_index]
    mean_byz = np.mean(byz_data, axis=0)

    """
    # Crear un gráfico de dispersión para mostrar los marcadores
    for client_index, client_data in enumerate(data):
        color = 'red' if client_index in undetected_byz_index else 'blue'
        if client_index in undetected_byz_index:
            plt.scatter(iter, client_data, label='Bizantino #{}'.format(client_index + 1), color=color, marker='o')
    """

    # Representar la media de los clientes benignos
    plt.scatter(iter, mean_benign, label='Media dos benignos', color='green', marker='o')
    plt.scatter(iter, mean_byz, label='Media dos byzantinos', color='red', marker='x')

    # Añadir una línea constante en la iteración 11
    if detectar:
        plt.axvline(x=11 + FLDET_START, color='gray', linestyle='dotted', linewidth=1.5, label='Comezo de deteccións')

    plt.xlabel('Iteración')
    plt.ylabel('Malicious Score')
    plt.title('Evolución das Malicious Scores - {}'.format(type))

    # Añadir leyenda manualmente
    plt.legend(loc='upper left')

    if gardar:
        plt.savefig(save_path + '/medias.png')
    plt.show()


def grafica_cluster(mal_scores, undetected_byz_index, epoch=None):
    # Colorear según si es byzantino o no
    colores = ['red' if i in undetected_byz_index else 'blue' for i in range(len(mal_scores))]
    last_iter = [client[-1] for client in mal_scores]
    plt.scatter(range(len(mal_scores)), last_iter, color=colores)

    # Añadir leyenda manualmente
    clientes_bizantinos = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10)
    clientes_benignos = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10)
    plt.legend([clientes_bizantinos, clientes_benignos], ['Clientes byzantinos', 'Clientes benignos'])

    # Dibujar el ndarray en un gráfico
    plt.xlabel('Índice do cliente')
    plt.ylabel('Malicious score')
    if epoch is None:
        plt.title('Puntuacións maliciosas na iteración da clusterización')
    else:
        plt.title(f'Puntuacións maliciosas na iteración {epoch}')

    if gardar:
        plt.savefig(save_path + '/cluster.png')
    plt.show()


def debuxar_precisions(datos, attack_type):
    # Asignar nombres a las columnas
    datos.columns = ["Iteracion", "ACC_Global"]

    # Crear la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(datos['Iteracion'], datos['ACC_Global'], marker='^', label='ACC Global')

    # Configurar la gráfica
    plt.title('Precisión ao longo do adestramento - {}'.format(attack_type))
    plt.xlabel('Iteraciones')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid(True)

    if gardar:
        plt.savefig(save_path + '/precisions.png')
    plt.show()


def debuxar_precisions_sen_ataque(datos):
    # Asignar nombres a las columnas
    datos.columns = ["Iteracion", "Media_ACC_Benigno", "Media_ACC_Byzantino", "ACC_Global"]

    # Crear la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(datos['Iteracion'], datos['Media_ACC_Benigno'], marker='o', label='Media ACC Benigno')
    plt.plot(datos['Iteracion'], datos['ACC_Global'], marker='^', label='ACC Global')

    # Configurar la gráfica
    plt.title('Precisión ao longo do adestramento - {}'.format(attack_type))
    plt.xlabel('Iteraciones')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid(True)

    if gardar:
        path_gardar = path + '/precisions.png'
        plt.savefig(path_gardar)
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

    path = 'PROBAS/ND/20240321-231208/no/simple_mean/0'
    save_path = os.path.dirname(path)
    attack_type = path.split('/')[-3]
    # leer desde la ruta
    data = pd.read_csv(path + '/score.csv', header=None)
    data = data.T.values.tolist()
    data2 = pd.read_csv(path + '/acc.csv', header=0)
    undetected = leer_undetected(path + '/Ataques_Detectados.txt')
    if undetected is not None:
        debuxar_medias(data, attack_type, undetected)
        grafica_cluster(data, undetected)
        debuxar_precisions(data2, attack_type)
    else:
        debuxar_precisions_sen_ataque(data2)
