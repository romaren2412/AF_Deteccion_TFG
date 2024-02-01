import re
import matplotlib.pyplot as plt
import numpy as np

def grafica(ruta_carpeta):
    if ruta_carpeta.__contains__('Silueta_00'):
        return None
    ruta_archivo = ruta_carpeta + '/Resumo.txt'
    # Leer el contenido del archivo
    with open(ruta_archivo, 'r') as archivo:
        texto = archivo.read()

    # Expresiones regulares para encontrar valores
    regex_silueta_actual = r"Silueta actual: (\d+.\d+)"
    regex_erroneas_actual = r"Foron evitadas (\d+) \(\d+.\d+%\) eliminacións erróneas"
    regex_acertadas_actual = r"Foron evitadas (\d+) \(\d+.\d+%\) eliminacións acertadas"
    regex_silueta = r"No caso de establecer unha silueta de (\d+.\d+):"
    regex_erroneas = r"Evitaríanse (\d+) \(\d+.\d+%\) eliminacións erróneas"
    regex_acertadas = r"Evitaríanse (\d+) \(\d+.\d+%\) eliminacións acertadas"

    # Buscar coincidencias usando expresiones regulares
    silueta_actual_match = re.search(regex_silueta_actual, texto)
    silueta_actual = float(silueta_actual_match.group(1)) if silueta_actual_match else None
    erroneas_actual_match = re.search(regex_erroneas_actual, texto)
    erroneas_actual = int(erroneas_actual_match.group(1)) if erroneas_actual_match else None
    acertadas_actual_match = re.search(regex_acertadas_actual, texto)
    acertadas_actual = int(acertadas_actual_match.group(1)) if acertadas_actual_match else None

    silueta_matches = re.findall(regex_silueta, texto)
    erroneas_matches = re.findall(regex_erroneas, texto)
    acertadas_matches = re.findall(regex_acertadas, texto)

    # Convertir las cadenas extraídas a números enteros
    siluetas = [silueta_actual] + [float(match) for match in silueta_matches]
    erroneas_values = [erroneas_actual] + [int(match) for match in erroneas_matches]
    acertadas_values = [acertadas_actual] + [int(match) for match in acertadas_matches]

    # Configurar posiciones de barras utilizando NumPy
    ind = np.arange(len(siluetas))
    width = 0.35

    # Crear la gráfica de barras
    plt.bar(ind - width/2, erroneas_values, width, label='Erros evitados')
    plt.bar(ind + width/2, acertadas_values, width, label='Acertos evitados', alpha=0.5)

    # Configurar etiquetas y leyenda
    plt.xlabel('Valor de Silueta')
    # Configurar ticks en el eje Y
    max_value = max(erroneas_values + acertadas_values) + 1
    y_ticks = np.arange(0, max_value, max(1, max_value // 10))  # Ajustar el divisor para controlar la cantidad de ticks
    plt.yticks(y_ticks)
    plt.xticks(ind, siluetas)  # Establecer las etiquetas en el eje X
    plt.legend(loc='lower right')
    plt.title('Comparación de eliminacións erróneas e acertadas')
    plt.savefig(ruta_carpeta + '/EliminacionsEvitadas.png')
    # Mostrar la gráfica
    plt.show()

if __name__ == '__main__':
    # Ruta del archivo a procesar
    ruta_archivo = 'ProbaSilueta/Datos/Silueta_70/backdoor/simple_mean'

    # Procesar el archivo
    grafica(ruta_archivo)
