import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

FLDET_START = 50


def debuxar_precision(datos):
    # Asignar nombres a las columnas
    datos.columns = ["Iteracion", "ACC_1e1", "ACC_2e1", "ACC_5e1", "ACC_1", "ACC_1e3", "ACC_1e2"]

    # Crear la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(datos['Iteracion'], datos['ACC_1e3'], label='LR=0.001')
    plt.plot(datos['Iteracion'], datos['ACC_1e2'], label='LR=0.01')
    plt.plot(datos['Iteracion'], datos['ACC_1e1'], label='LR=0.1')
    plt.plot(datos['Iteracion'], datos['ACC_2e1'], label='LR=0.2')
    plt.plot(datos['Iteracion'], datos['ACC_5e1'], label='LR=0.5')
    plt.plot(datos['Iteracion'], datos['ACC_1'], label='LR=1')

    # Configurar la gráfica
    plt.title('Evolución da precisión según a taxa de aprendizaxe')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid(True)

    if gardar:
        plt.savefig(save_path + 'precisions.png')
    plt.show()


if __name__ == "__main__":
    gardar = True
    detectar = False

    path = '.'
    save_path = os.path.dirname(path)
    data_acc = pd.read_csv(path + '/Accuracies.csv', header=0)
    debuxar_precision(data_acc)
