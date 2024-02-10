import re
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

class CasoSilueta:
    def __init__(self, silueta, erroneas, acertadas):
        self.silueta = silueta
        self.erroneas = erroneas
        self.acertadas = acertadas

def obtener_casos_silueta_desde_csv(ruta_csv):
    with open(ruta_csv, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Saltar la primera fila que contiene los nombres de las columnas
        casos_silueta = [CasoSilueta(float(row[0].split()[1]) / 100, int(row[1]), int(row[2])) for row in reader]
    return casos_silueta

def grafica_desde_csv(ruta_carpeta, ruta_csv):
    if ruta_carpeta.__contains__('Silueta_00'):
        return None

    instancias_casos_silueta = obtener_casos_silueta_desde_csv(ruta_csv)

    # Configurar posiciones de barras utilizando NumPy
    ind = np.arange(len(instancias_casos_silueta))
    width = 0.35

    # Crear la gráfica de barras
    erroneas_values = [caso_silueta.erroneas for caso_silueta in instancias_casos_silueta]
    acertadas_values = [caso_silueta.acertadas for caso_silueta in instancias_casos_silueta]

    plt.bar(ind - width / 2, erroneas_values, width, label='Erros evitados')
    plt.bar(ind + width / 2, acertadas_values, width, label='Acertos evitados', alpha=0.5)

    # Configurar etiquetas y leyenda
    plt.xlabel('Valor de Silueta')
    max_value = max(max(erroneas_values), max(acertadas_values)) + 1
    y_ticks = np.arange(0, max_value, max(1, max_value // 10))
    plt.yticks(y_ticks)
    plt.xticks(ind, [caso_silueta.silueta for caso_silueta in instancias_casos_silueta])

    plt.legend(loc='lower right')
    plt.title('Comparación de eliminacións erróneas e acertadas')
    plt.savefig(ruta_carpeta + f"/{ataque}.png")
    plt.show()

if __name__ == '__main__':
    # Lista de carpetas de siluetas
    carpetas_siluetas = ['Silueta_70', 'Silueta_80', 'Silueta_90']

    # Lista de opciones de subcarpetas
    subcarpetas_opciones = ['backdoor', 'dba']  # Puedes agregar más opciones según sea necesario

    # Lista de ataques y agregaciones
    ataques = ['backdoor', 'full_mean_attack', 'full_trim', 'dba', 'partial_trim']
    agregaciones = ['median', 'trim', 'simple_mean']

    for ataque in ataques:
        for agregacion in agregaciones:
            try:
                # Obtener la ruta del archivo CSV
                ruta_csv = f'CSV_Siluetas/{ataque}_{agregacion}.csv'

                # Procesar el archivo CSV y crear la gráfica
                ruta_carpeta = os.path.join('CSV_Siluetas/Graficas', agregacion)
                if not os.path.exists(ruta_carpeta):
                    os.makedirs(ruta_carpeta)
                grafica_desde_csv(ruta_carpeta, ruta_csv)

            except Exception as e:
                print(f"An error occurred while processing {ataque} with {agregacion}: {str(e)}")

