import os
import re
import time

import numpy as np
from matplotlib import pyplot as plt

from Grafica_Resumo import grafica


class Ataque:
    def __init__(self, acc, recall, silhouette, detected_clients):
        self.acc = acc
        self.recall = recall
        self.silhouette = silhouette
        self.detected_clients = detected_clients

    def __str__(self):
        return f"Acc: {self.acc}, Recall: {self.recall}, Silhouette: {self.silhouette}, Detected clients: {self.detected_clients}"


class Entrenamiento:
    def __init__(self, benignos, bizantinos, porcentaje_benignos, ataques):
        self.benignos = benignos
        self.bizantinos = bizantinos
        self.porcentaje_benignos = porcentaje_benignos
        self.ataques = ataques
        self.precision = None

    def __str__(self):
        return f"Benignos: {self.benignos}, Bizantinos: {self.bizantinos}, Porcentaje de benignos: {self.porcentaje_benignos}, Ataques: {self.ataques}"

    def setAtaques(self, ataques):
        self.ataques = ataques

    def setPrecision(self, precision):
        self.precision = precision

    # getters
    def getBenignos(self):
        return self.benignos

    def getBizantinos(self):
        return self.bizantinos

    def getPorcentajeBenignos(self):
        return self.porcentaje_benignos

    def getAtaques(self):
        if self.ataques is None:
            return []
        return self.ataques

    def getPrecision(self):
        return self.precision


def contar_subcarpetas(ruta):
    # Obtener la lista de elementos en la carpeta
    elementos = os.listdir(ruta)
    # Filtrar solo las carpetas
    subcarpetas = [elemento for elemento in elementos if os.path.isdir(os.path.join(ruta, elemento))]

    # Contar y mostrar el resultado
    return len(subcarpetas)


def leer_archivo_precision(ruta_archivo):
    try:
        with open(ruta_archivo, 'r') as archivo:
            lineas = archivo.readlines()

            # Extraer el último valor de la lista
            if lineas:
                ultimo_valor = float(lineas[-1].strip())
                return ultimo_valor
            else:
                print(f"El archivo '{ruta_archivo}' está vacío.")
    except FileNotFoundError:
        print(f"Error: El archivo '{ruta_archivo}' no existe.")
    except ValueError:
        print(f"Error: No se pudo convertir el último valor a un número en el archivo '{ruta_archivo}'.")
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")

    # Devolver un valor predeterminado en caso de error
    return None


def leer_archivo_txt(ruta_carpeta):
    ruta_Ataques = ruta_carpeta + '/Ataques_Detectados.txt'
    ruta_Precision = ruta_carpeta + '/Precision.txt'
    # Inicializar variables para almacenar los valores
    benignos = []
    bizantinos = []
    acc = []
    detected_clients = []
    silhouette = []
    porcentaje_benignos = None

    try:
        with open(ruta_Ataques, 'r') as archivo:
            contenido = archivo.read()

            # Extraer valores usando expresiones regulares
            match_benignos = re.search(r'Benignos: (\[.*\])', contenido)
            if match_benignos:
                benignos = eval(match_benignos.group(1))

            match_bizantinos = re.search(r'Byzantinos.*?: (\[.*\])', contenido)
            if match_bizantinos:
                bizantinos = eval(match_bizantinos.group(1))

            match_porcentaje_benignos = re.search(r'Porcentaxe de benignos restantes: ([\d.]+)', contenido)
            if match_porcentaje_benignos:
                porcentaje_benignos = float(match_porcentaje_benignos.group(1))

            entrenamiento = Entrenamiento(benignos, bizantinos, porcentaje_benignos, None)

            # Ler os ataques detectados
            if "------" in contenido:
                match_acc = re.findall(r'acc (.*?);', contenido)
                acc = [float(x) for x in match_acc]

                match_recall = re.findall(r'recall (.*?);', contenido)
                recall = [float(x) for x in match_recall]

                match_detected_clients = re.findall(r'detected_clients: (\[.*\])', contenido)
                detected_clients = [eval(x) for x in match_detected_clients]

                match_silhouette = re.findall(r'silhouette:\s*([\d.]+)', contenido, re.MULTILINE)
                silhouette = [float(x) for x in match_silhouette]

                # Crear el objeto Entrenamiento
                ataques = [Ataque(acc[i], recall[i], silhouette[i], detected_clients[i]) for i in range(len(acc))]
                entrenamiento.setAtaques(ataques)
            else:
                print("O adestramento finalizou sen detectar ningún ataque.")
                precision = leer_archivo_precision(ruta_Precision)
                entrenamiento.setPrecision(precision)

            # Devolver los valores extraídos
            return entrenamiento

    except FileNotFoundError:
        print("Error: El archivo no existe.")
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")

    # Devolver valores predeterminados en caso de error
    return None


def debuxar_porcentaxes(entrenamientos, bizantinos_inic):
    plt.clf()

    bizantinos = []
    benignos = []
    for e in entrenamientos:
        bizantinos.append(len(e.getBizantinos()) / len(bizantinos_inic))
        benignos.append(e.getPorcentajeBenignos())

    plt.plot(bizantinos, label='Bizantinos', linestyle='-', color='red', marker='o')
    plt.plot(benignos, label='Benignos', linestyle='--', color='green', marker='o')
    # Añadir etiquetas y leyenda
    plt.xlabel('Entrenamento')
    plt.xticks(np.arange(0, len(bizantinos), 1))
    plt.title('Evolución de % de tipo de clientes')
    plt.legend()
    plt.savefig(ruta_carpeta + '/EvolucionClientes.png')
    #plt.show()


def debuxar_todo(entrenamientos, bizantinos_inic, silueta, marxe):
    plt.clf()

    silhouettes = []
    biz_percent = []
    ben_percent = []
    clasificacion_ataques = []

    # Calcular datos para debuxar
    for e in entrenamientos:
        # De cada entreno, almacenar os bizantinos e o % de benignos
        bizantinos = e.getBizantinos()
        porcentaje = e.getPorcentajeBenignos()
        # De cada ataque do entreno:
        # 1. Almacenar a silueta e clasificar o ataque en función da mesma
        # 2. Gardar o % de bizantinos e benignos
        for a in e.getAtaques():
            silhouettes.append(a.silhouette)
            # Clasificar
            if (a.acc > 1 - marxe) and (a.recall != -1):
                clasificacion_ataques.append(1)
            elif a.silhouette < umbral_silueta:
                clasificacion_ataques.append(2)
            else:
                clasificacion_ataques.append(0)

            biz_percent.append(len(bizantinos) / len(bizantinos_inic))
            ben_percent.append(porcentaje)

        # Se non hai ataques, engadir os valores do adestramento
        if not e.getAtaques():
            biz_percent.append(len(bizantinos) / len(bizantinos_inic))
            ben_percent.append(porcentaje)

    # Repetir o último valor ata o final
    while len(biz_percent) < len(silhouettes) + 2:
        biz_percent.append(biz_percent[-1])
        ben_percent.append(ben_percent[-1])

    # Colores para las barras
    colores = ['lightgrey' if value == 0 else 'lightgreen' if value == 2 else 'lightblue' for value in
               clasificacion_ataques]
    plt.figure(figsize=(9, 6))
    plt.plot(biz_percent, label='% Bizantinos', linestyle='-', color='red', marker='o')
    plt.plot(ben_percent, label='% Benignos', linestyle='--', color='green', marker='o')
    plt.bar(range(1, len(silhouettes) + 1), silhouettes, color=colores, width=0.15)

    if silueta != 0:
        # Modificar manualmente la leyenda
        legend_labels = {'lightgrey': 'Eliminación incorrecta', 'lightblue': 'Eliminación acertada',
                         'lightgreen': 'Evitado por silueta'}
        plt.legend([plt.Line2D([0], [0], color='red', linestyle='-', marker='o'),
                    plt.Line2D([0], [0], color='green', linestyle='--', marker='o')] + [
                       plt.Rectangle((0, 0), 1, 1, color=color) for color in legend_labels.keys()] +
                   [plt.Line2D([0], [0], color='b', linestyle='dotted', label='Límite silueta')]
                   , ['% Bizantinos', '% Benignos'] + list(legend_labels.values()) + ['Límite silueta'],
                   fontsize='small')
        # Añadir etiquetas y leyenda
        plt.axhline(y=silueta, color='b', linestyle='dotted', label='Límite silueta')
    else:
        # Modificar manualmente la leyenda
        legend_labels = {'lightgrey': 'Silueta (erro)', 'lightblue': 'Silueta (acerto)'}
        plt.legend([plt.Line2D([0], [0], color='red', linestyle='-', marker='o'),
                    plt.Line2D([0], [0], color='green', linestyle='--', marker='o')] + [
                       plt.Rectangle((0, 0), 1, 1, color=color) for color in legend_labels.keys()]
                   , ['% Bizantinos', '% Benignos'] + list(legend_labels.values()),
                   fontsize='small')

    plt.xlabel('Ataque identificado')
    # Configurar las etiquetas del eje x
    etiquetas = ['START'] + [str(i) for i in range(1, len(silhouettes) + 1)] + ['END']
    plt.xticks(np.arange(0, len(silhouettes) + 2, 1), etiquetas)
    # plt.xticks(np.arange(0, len(silhouettes), 1))
    plt.title('Evolución de % de tipo de clientes e siluetas')
    plt.savefig(ruta_carpeta + '/EvolucionSiluetas.png')
    # plt.show()


class CasoSilueta:
    def __init__(self, silueta):
        self.silueta = silueta
        self.erros = 0
        self.acertos = 0

    def calculoErros(self, accuracies, silhouettes, marxe, recall):
        for i in range(len(accuracies)):
            # Eliminacións erróneas "evitables"
            if accuracies[i] < (1 - marxe) and silhouettes[i] < self.silueta:
                self.erros += 1
            # Eliminacións acertadas "evitables"
            elif accuracies[i] > (1 - marxe) and silhouettes[i] < self.silueta and recall[i] != -1:
                self.acertos += 1

    def imprimirInfo(self, silueta_real, accuracies, f):
        if self.silueta == silueta_real:
            f.write(f"--------\nSILUETA ACTUAL: {umbral_silueta}\n")
            f.write(
                f"Foron evitadas {self.erros} ({self.erros / len(accuracies) * 100}%) eliminacións erróneas\n")
            f.write(
                f"Foron evitadas {self.acertos} ({self.acertos / len(accuracies) * 100}%) eliminacións acertadas\n")
            f.write("--------------------\n\n")
        else:
            f.write(f"No caso de establecer unha silueta de {self.silueta}:\n")
            f.write(
                f"Evitaríanse {self.erros} ({self.erros / len(accuracies) * 100}%) eliminacións erróneas\n")
            f.write(f"Evitaríanse {self.acertos} ({self.acertos / len(accuracies) * 100}%)"
                    f" eliminacións acertadas\n")
            f.write("--------------------\n\n")


def obter_datos(entrenamientos, marxe):
    # Calcular % de ataques evitables
    # Coa silueta actual pódense observar 4 posibilidades:
    #   - Eliminacións erróneas evitadas (se silueta é distinto de 0)
    #   - Eliminacións acertadas evitadas (se silueta é distinto de 0)
    #   - Eliminacións erróneas evitables (observar eliminacións erróneas cunha silueta maior)
    #   - Eliminacións acertadas evitables (observar eliminacións acertadas cunha silueta maior)
    accuracies = []
    silhouettes = []
    recall = []

    # Cálculo de accuracies e silhouettes
    for e in entrenamientos:
        for a in e.getAtaques():
            accuracies.append(a.acc)
            silhouettes.append(a.silhouette)
            recall.append(a.recall)

    siluetas_opcionais = [i / 100 for i in range(70, 100, 5)]
    if umbral_silueta not in siluetas_opcionais:
        siluetas_opcionais.append(umbral_silueta)
        siluetas_opcionais.sort()

    with open(ruta_carpeta + '/Resumo.txt', 'w') as f:
        # COA SILUETA ACTUAL
        for i in range(len(accuracies)):
            f.write(f"Ataque {i} (acc: {accuracies[i]}, silhouette {silhouettes[i]})\n")

        f.write("\n--------\n")

        for sil in siluetas_opcionais:
            s = CasoSilueta(sil)
            s.calculoErros(accuracies, silhouettes, marxe, recall)
            s.imprimirInfo(umbral_silueta, accuracies, f)


def probar(ruta_carpeta):
    # Llama a la función para contar subcarpetas
    n_subcarpetas = contar_subcarpetas(ruta_carpeta)
    marxe = 0.1
    entrenamientos = []

    for i in range(n_subcarpetas):
        ruta = ruta_carpeta + '/' + str(i)
        entrenamiento = leer_archivo_txt(ruta)
        # Calcular % bizantinos restantes para imprimir
        if i == 0:
            bizantinos_inic = entrenamiento.getBizantinos()
        print("Adestramento " + str(i) + ":")
        print("Benignos: ", entrenamiento.getBenignos())
        print("Byzantinos: ", entrenamiento.getBizantinos())
        print("Porcentaje de bizantinos restantes: ", len(entrenamiento.getBizantinos()) / len(bizantinos_inic))
        print("Porcentaje de benignos restantes: ", entrenamiento.getPorcentajeBenignos())
        entrenamientos.append(entrenamiento)

    debuxar_porcentaxes(entrenamientos, bizantinos_inic)
    debuxar_todo(entrenamientos, bizantinos_inic, umbral_silueta, marxe)
    obter_datos(entrenamientos, marxe)


if __name__ == "__main__":
    # Ingresa la ruta de la carpeta que quieres analizar
    ruta_global = 'ProbaSilueta/Datos_v3/'
    subcarpetas_Silueta = [e for e in os.listdir(ruta_global) if os.path.isdir(os.path.join(ruta_global, e))]
    for silueta_XX in subcarpetas_Silueta:
        umbral_silueta = float(silueta_XX.split('_')[1]) / 100
        for ataque in os.listdir(ruta_global + silueta_XX):
            for aggr in ['median', 'trim', 'simple_mean']:
                ruta_carpeta = ruta_global + silueta_XX + '/' + ataque + '/' + aggr
                print(f"Probando {ruta_carpeta}...")
                if os.path.exists(ruta_carpeta):
                    if os.path.exists(ruta_carpeta + '/EvolucionClientes.png'):
                        os.remove(ruta_carpeta + '/EvolucionClientes.png')
                    if os.path.exists(ruta_carpeta + '/EvolucionSiluetas.png'):
                        os.remove(ruta_carpeta + '/EvolucionSiluetas.png')
                    probar(ruta_carpeta)
                    grafica(ruta_carpeta)
