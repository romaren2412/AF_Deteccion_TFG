import os
import csv

# Carpeta base donde se encuentran las carpetas siluetaXX
carpeta_base = 'ProbaSilueta/Datos_v3_TEMP'

# Lista de carpetas de siluetas
carpetas_siluetas = ['Silueta_70', 'Silueta_80', 'Silueta_90']

# Posibilidades
ataques = ['backdoor', 'full_mean_attack', 'full_trim', 'dba', 'partial_trim']
agregaciones = ['median', 'trim', 'simple_mean']


# Función para procesar un archivo de valores.txt y obtener datos
def procesar_archivo_valores(ruta_archivo):
    with open(ruta_archivo, 'r') as file:
        lineas = file.readlines()

    # Asumimos que las líneas siguen el formato proporcionado
    eliminaciones_erroneas = int(lineas[0].split()[0])
    eliminaciones_acertadas = int(lineas[1].split()[0])

    return eliminaciones_erroneas, eliminaciones_acertadas


# Función para procesar todas las carpetas de siluetas
def procesar_siluetas(carpeta_base, carpetas_siluetas, ataque, agregacion):
    datos_totales = []

    for silueta in carpetas_siluetas:
        ruta_carpeta_silueta = os.path.join(carpeta_base, silueta, ataque, agregacion)
        ruta_archivo_valores = os.path.join(ruta_carpeta_silueta, 'valores.txt')

        if os.path.exists(ruta_archivo_valores):
            datos = procesar_archivo_valores(ruta_archivo_valores)
            datos_totales.append(datos)
        else:
            print(f"El archivo valores.txt no existe en {ruta_carpeta_silueta}")

    return datos_totales


# Función para escribir en un archivo CSV
def escribir_csv(nombre_archivo, datos):
    filas = ['silueta 70', 'silueta 80', 'silueta 90']
    columnas = ['Eliminaciones Erróneas', 'Eliminaciones Acertadas']

    with open(nombre_archivo, 'w', newline='') as file:
        writer = csv.writer(file)
        # Escribir la primera fila con los nombres de las columnas
        writer.writerow([''] + columnas)
        # Escribir los datos para cada fila y columna
        for i, fila in enumerate(filas):
            writer.writerow([fila] + list(datos[i]))

for ataque in ataques:
    for agregacion in agregaciones:
        try:
            datos_totales = procesar_siluetas(carpeta_base, carpetas_siluetas, ataque, agregacion)
            nombre_archivo_csv = f'CSV_Siluetas/{ataque}_{agregacion}.csv'
            escribir_csv(nombre_archivo_csv, datos_totales)
        except Exception as e:
            print(f"An error occurred while processing {ataque} with {agregacion}: {str(e)}")
            os.remove(nombre_archivo_csv)
