import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from config import parse_args

args = parse_args()


def cargar_datos(ruta):
    """ Carga las puntuaciones y los encabezados desde un archivo CSV """
    data = pd.read_csv(ruta)
    headers = data.columns.tolist()  # Obtiene los encabezados
    scores = data.values  # Todas las filas de datos
    return headers, scores


def calcular_medias_por_epoca(ruta):
    headers, scores = cargar_datos(ruta)
    """Calcula las medias de los clientes benignos y bizantinos para cada época"""
    num_epochs = scores.shape[0]
    num_clients = scores.shape[1]

    benign_means = []
    byzantine_means = []

    for epoch in range(num_epochs):
        benign_scores = [scores[epoch, i] for i in range(num_clients) if 'Ben' in headers[i]]
        byzantine_scores = [scores[epoch, i] for i in range(num_clients) if 'Byz' in headers[i]]

        benign_means.append(np.mean(benign_scores) if benign_scores else 0)
        byzantine_means.append(np.mean(byzantine_scores) if byzantine_scores else 0)

    return benign_means, byzantine_means


def plot_media_por_cliente(ruta):
    headers, scores = cargar_datos(ruta)
    """ Gráfica A: Muestra la media de cada cliente en un gráfico de barras """
    mean_scores = np.mean(scores, axis=0)
    indices = np.arange(len(mean_scores))

    # Colores según benignos o bizantinos
    colors = ['red' if 'Byz' in header else 'green' for header in headers]

    plt.bar(indices, mean_scores, color=colors)
    plt.xlabel('Cliente')
    plt.ylabel('Media de Puntuación')
    # plt.yscale('log')
    plt.title(f'Media de Puntuacións por Cliente - {attack_type}')
    plt.xticks(indices, [f'#{index}' for index in indices])

    if gardar:
        plt.savefig(save_path + '/bar.png')
    plt.show()


def plot_evolucion_medias(ruta):
    """ Gráfica B: Muestra la evolución de las medias de los bizantinos y benignos a lo largo de las épocas """
    mean_benign, mean_byzantine = calcular_medias_por_epoca(ruta)

    if attack_type == 'no':
        plt.plot(mean_benign, label='Media de Benignos', color='green', marker='o')
    else:
        plt.plot(mean_benign, label='Media de Benignos', color='green', marker='o')
        plt.plot(mean_byzantine, label='Media de Bizantinos', color='red', marker='x')

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Época')
    plt.ylabel('Media de Puntuación')
    plt.title(f'Evolución de las Medias de Puntuaciones - {attack_type}')
    plt.legend()

    if gardar:
        plt.savefig(save_path + '/plot.png')
    plt.show()


def debuxar_diferencias_precision(ruta):
    # Cargar datos desde un archivo CSV
    data = pd.read_csv(ruta + '/acc_comp.csv')
    ataque = ruta.split('/')[-1]

    # Crear un gráfico de líneas para ACC_FedAvg y ACC_Flare
    plt.figure(figsize=(10, 6))
    plt.plot(data['Iteracions'], data['ACC_Global'], label='ACC_FedAvg', marker='o', linestyle='--', color='gray')
    plt.plot(data['Iteracions'], data['ACC_FLTrust'], label='ACC_FLTrust', marker='x', linestyle='-', color='blue')

    if ataque in ('backdoor', 'dba'):
        # Trazar también las métricas de ASR si es un backdoor
        plt.plot(data['Iteracions'], data['ASR'], label='ASR_FedAvg', linestyle='dotted',
                 color='gray')
        plt.plot(data['Iteracions'], data['ASR_FLTrust'], label='ASR_Flare', linestyle='dotted', color='blue')

    # Añadiendo título y etiquetas
    plt.title('FedAvg VS FLTrust - {}'.format(ataque))
    plt.xlabel('Iteracions')
    plt.ylabel('Precision')
    plt.legend()

    # Configurar el eje x para mostrar solo números enteros
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    if gardar:
        plt.savefig(ruta + '/plot.png')
    plt.show()


def debuxar_precision(ruta):
    # Cargar datos desde un archivo CSV
    datos = pd.read_csv(ruta)
    asr = False
    if len(datos.columns) == 3:
        asr = True
        datos.columns = ["Iteracion", "ACC_Global", "ASR"]
    else:
        datos.columns = ["Iteracion", "ACC_Global"]

    # Crear la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(datos['Iteracion'], datos['ACC_Global'], marker='^', label='ACC Global')
    if asr:
        plt.plot(datos['Iteracion'], datos['ASR'], marker='o', color='red', label='Éxito do ataque',
                 linestyle='dashed')

    # Configurar la gráfica
    plt.title('Evolución da precisión - {}'.format(attack_type))
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()

    if gardar:
        plt.savefig(save_path + '/precisions.png')
    plt.show()


if __name__ == "__main__":
    gardar = True

    path = 'PROBAS/ProbasDef/mean_attack'
    save_path = path
    attack_type = path.split('/')[-1]

    # plot_media_por_cliente(path + '/score.csv')
    # plot_evolucion_medias(path + '/score.csv')
    debuxar_diferencias_precision('PROBAS/FedAvg/no')
    # debuxar_precision(path + '/acc.csv')
