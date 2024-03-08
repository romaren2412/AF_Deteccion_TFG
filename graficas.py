import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def debuxar_medias(data, type, undetected_byz_index):
    iter = list(range(1, len(data[0]) + 1))

    # Calcular la media de los clientes benignos
    benign_data = [client_data for i, client_data in enumerate(data) if i not in undetected_byz_index]
    mean_benign = np.mean(benign_data, axis=0)

    # Calcular la media de los clientes malignos
    byz_data = [client_data for i, client_data in enumerate(data) if i in undetected_byz_index]
    mean_byz = np.mean(byz_data, axis=0)

    # Representar la media de los clientes benignos y malignos
    plt.plot(iter, mean_benign, label='Media dos benignos', color='green', linestyle='dashed', linewidth=2)
    plt.plot(iter, mean_byz, label='Media dos byzantinos', color='red', linestyle='dashed', linewidth=2)

    plt.xlabel('Iteración')
    plt.ylabel('Malicious Score')
    plt.title('Evolución das Malicious Scores - {}'.format(type))

    # Añadir leyenda manualmente
    plt.legend(loc='upper left')
    plt.show()


def grafica_cluster(mal_scores, undetected_byz_index):
    # Colorear según si es byzantino o no
    colores = ['red' if i in undetected_byz_index else 'blue' for i in range(len(mal_scores))]
    last_iter = [client[-1] for client in mal_scores]
    plt.scatter(range(len(mal_scores)), last_iter, color=colores)

    #Añadir leyenda manualmente
    clientes_bizantinos = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10)
    clientes_benignos = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10)
    plt.legend([clientes_bizantinos, clientes_benignos], ['Clientes byzantinos', 'Clientes benignos'])

    # Dibujar el ndarray en un gráfico
    plt.xlabel('Índice do cliente')
    plt.ylabel('Malicious score')
    plt.title('Puntuacións maliciosas na iteración da clusterización')
    plt.show()


def debuxar_precisions(datos, attack_type):
    # Asignar nombres a las columnas
    datos.columns = ["Iteracion", "Media_ACC_Benigno", "Media_ACC_Byzantino", "ACC_Global"]

    # Crear la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(datos['Iteracion'], datos['Media_ACC_Benigno'], marker='o', label='Media ACC Benigno')
    plt.plot(datos['Iteracion'], datos['Media_ACC_Byzantino'], marker='s', label='Media ACC Byzantino')
    plt.plot(datos['Iteracion'], datos['ACC_Global'], marker='^', label='ACC Global')

    # Configurar la gráfica
    plt.title('Precisión ao longo do adestramento - {}'.format(attack_type))
    plt.xlabel('Iteraciones')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid(True)

    # Mostrar la gráfica
    plt.show()


if __name__ == "__main__":
    path = 'PROBAS/20240308-190908/backdoor/simple_mean/0'
    attack_type = path.split('/')[-3]
    # leer desde la ruta
    data = pd.read_csv(path + '/score1.csv', header=None)
    data = data.T.values.tolist()
    undetected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    debuxar_medias(data, attack_type, undetected)
    grafica_cluster(data, undetected)

    data2 = pd.read_csv(path + '/acc.csv', header=0)

    debuxar_precisions(data2, attack_type)
