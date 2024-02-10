import re
import matplotlib.pyplot as plt
import numpy as np

class CasoSilueta:
    def __init__(self, silueta, erroneas, acertadas):
        self.silueta = silueta
        self.erroneas = erroneas
        self.acertadas = acertadas

def obtener_casos_silueta(texto):
    regex_silueta_actual = r"SILUETA ACTUAL: (\d+.\d+)"
    regex_erroneas_actual = r"Foron evitadas (\d+) \(\d+.\d+%\) eliminacións erróneas"
    regex_acertadas_actual = r"Foron evitadas (\d+) \(\d+.\d+%\) eliminacións acertadas"

    regex_silueta = r"No caso de establecer unha silueta de (\d+.\d+):"
    regex_erroneas = r"Evitaríanse (\d+) \(\d+.\d+%\) eliminacións erróneas"
    regex_acertadas = r"Evitaríanse (\d+) \(\d+.\d+%\) eliminacións acertadas"

    silueta_actual_match = re.search(regex_silueta_actual, texto)
    silueta_actual = float(silueta_actual_match.group(1)) if silueta_actual_match else None
    erroneas_actual_match = re.search(regex_erroneas_actual, texto)
    erroneas_actual = int(erroneas_actual_match.group(1)) if erroneas_actual_match else None
    acertadas_actual_match = re.search(regex_acertadas_actual, texto)
    acertadas_actual = int(acertadas_actual_match.group(1)) if acertadas_actual_match else None

    silueta_matches = re.findall(regex_silueta, texto)
    erroneas_matches = re.findall(regex_erroneas, texto)
    acertadas_matches = re.findall(regex_acertadas, texto)

    casos_silueta = [CasoSilueta(silueta_actual, erroneas_actual, acertadas_actual)]

    for match in zip(silueta_matches, erroneas_matches, acertadas_matches):
        silueta, erroneas, acertadas = map(float, match)
        casos_silueta.append(CasoSilueta(silueta, int(erroneas), int(acertadas)))

    # Ordenar os casos por silueta
    casos_silueta.sort(key=lambda caso: caso.silueta)

    return casos_silueta

def grafica(ruta_carpeta):
    if ruta_carpeta.__contains__('Silueta_00'):
        return None
    ruta_archivo = ruta_carpeta + '/Resumo.txt'
    # Leer el contenido del archivo
    with open(ruta_archivo, 'r') as archivo:
        texto = archivo.read()

    instancias_casos_silueta = obtener_casos_silueta(texto)

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
    plt.savefig(ruta_carpeta + "/EliminacionsEvitadas.png")
    #plt.show()

if __name__ == '__main__':
    # Ruta del archivo a procesar
    ruta_archivo = 'ProbaSiluetav2/Silueta_90/backdoor/median'

    # Procesar el archivo
    grafica(ruta_archivo)
