# coding: utf-8
import matplotlib.pyplot as plt
import pandas as pd

FLDET_START = 50


def debuxar_precision_cinco(datos):
    # Asignar nombres a las columnas
    datos.columns = ['Iteracion', 'ACC_MNIST', 'ACC_MNISTM', 'ACC_SVHN', 'ACC_SYN', 'ACC_USPS', 'ACC_Global']

    # Crear la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(datos['Iteracion'], datos['ACC_MNIST'], label='MNIST', linestyle='dashed', marker='o')
    plt.plot(datos['Iteracion'], datos['ACC_MNISTM'], label='MNISTM', linestyle='dashed', marker='o')
    plt.plot(datos['Iteracion'], datos['ACC_SVHN'], label='SVHN', linestyle='dashed', marker='o')
    plt.plot(datos['Iteracion'], datos['ACC_SYN'], label='SYN', linestyle='dashed', marker='o')
    plt.plot(datos['Iteracion'], datos['ACC_USPS'], label='USPS', linestyle='dashed', marker='o')
    plt.plot(datos['Iteracion'], datos['ACC_Global'], label='Global', marker='o', linewidth=2, color='black')

    # Configurar la gráfica
    plt.title('Evolución da precisión - Digit Five')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid(True)

    if gardar:
        plt.savefig(save_path + 'precisions.png')
    plt.show()


def debuxar_medias_diferentes(data):
    it = list(range(1, len(data) + 1))
    # Ajustar los valores de los ticks sumándoles 50
    it = [tick + FLDET_START for tick in it]
    data.columns = ["MNIST", "MNISTM", "SVHN", "SYN", "USPS"]


    # Crear un gráfico de dispersión para mostrar los marcadores
    plt.scatter(it, data['MNIST'], label='MNIST', marker='x', linewidths=0.5)
    plt.scatter(it, data['MNISTM'], label='MNISTM', marker='x', linewidths=0.5)
    plt.scatter(it, data['SVHN'], label='SVHN', marker='x', linewidths=0.5)
    plt.scatter(it, data['SYN'], label='SYN', marker='x', linewidths=0.5)
    plt.scatter(it, data['USPS'], label='USPS', marker='x', linewidths=0.5)

    if detectar:
        plt.axvline(x=11 + FLDET_START, color='gray', linestyle='dotted', linewidth=1.5, label='Comezo de deteccións')

    plt.xlabel('Iteración')
    plt.ylabel('Malicious Score')
    plt.title(f'Evolución das puntuacións maliciosas - 4 tipos de datos diferentes')
    plt.legend(loc='upper left')

    if gardar:
        plt.savefig(save_path + '/medias.png')
    plt.show()


if __name__ == "__main__":
    gardar = True
    detectar = False

    path = '.'
    save_path = path
    data_acc = pd.read_csv(path + '/acc.csv', header=0)
    debuxar_precision_cinco(data_acc)
    data_score = pd.read_csv(path + '/score.csv', header=0)
    debuxar_medias_diferentes(data_score)
