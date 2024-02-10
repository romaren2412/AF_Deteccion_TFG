import os
import shutil

def copiar_imagenes(ruta_origen, ruta_destino):
    # Verificar si la ruta de destino existe, si no, crearla
    if not os.path.exists(ruta_destino):
        os.makedirs(ruta_destino)

    # Recorrer todas las carpetas y subcarpetas en la ruta de origen
    for carpeta_raiz, carpetas, archivos in os.walk(ruta_origen):
        for archivo in archivos:
            # Verificar si el archivo es "EliminacionsEvitadas.png"
            if archivo == "EliminacionsEvitadas.png":
                # Construir la ruta completa del archivo de origen
                ruta_origen_completa = os.path.join(carpeta_raiz, archivo)

                # Construir el nuevo nombre del archivo con la ruta
                nuevo_nombre = carpeta_raiz.replace(os.path.sep, "_") + ".png"

                # Construir la ruta completa del archivo de destino
                ruta_destino_completa = os.path.join(ruta_destino, nuevo_nombre)

                # Copiar el archivo a la nueva ubicación
                shutil.copy(ruta_origen_completa, ruta_destino_completa)

                print(f"Copiando {archivo} a {ruta_destino_completa}")

# Especifica las rutas de origen y destino
ruta_origen = r'ProbaSilueta/Datos_v2'
ruta_destino = r'Imaxes_v2'

# Llama a la función para copiar las imágenes
copiar_imagenes(ruta_origen, ruta_destino)
