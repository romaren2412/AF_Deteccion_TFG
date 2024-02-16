# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ruta_base = 'PROBAS/20240210-171339/ProbaLBFGS/full_mean_attack/simple_mean/0'

# Nombres de los archivos CSV
lbfgs_gitHub = ruta_base + '/hvp.csv'
lbfgs_fedRec = ruta_base + '/hvpAux.csv'

# Leer datos desde los archivos CSV
datos1 = pd.read_csv(lbfgs_gitHub)
datos2 = pd.read_csv(lbfgs_fedRec)

# Extraer columnas de media, norma y tiempo para cada archivo
media1, norma1, tiempo1 = datos1.iloc[:, 0], datos1.iloc[:, 1], datos1.iloc[:, 2]
media2, norma2, tiempo2 = datos2.iloc[:, 0], datos2.iloc[:, 1], datos2.iloc[:, 2]

print((tiempo1.mean() - tiempo2.mean()) / tiempo1.mean() * 100)


# Crear gráficas comparativas
plt.figure(figsize=(12, 6))

# Gráfica comparativa de medias
plt.subplot(1, 3, 1)
plt.plot(media1, label='GitHub', linestyle='--', marker='o')
plt.plot(media2, label='FedRec', linestyle='-', marker='x')
plt.xlabel('Índice do HVP')
plt.ylabel('Valor da media')
plt.legend()

# Gráfica comparativa de normas
plt.subplot(1, 3, 2)
plt.plot(norma1, label='GitHub', color='orange', linestyle='--', marker='o')
plt.plot(norma2, label='FedRec', color='green', linestyle='-', marker='x')
plt.xlabel('Índice do HVP')
plt.ylabel('Valor da norma')
plt.legend()

# Gráfica de barras para tiempos
plt.subplot(1, 3, 3)
bar_width = 0.35
bar_positions1 = np.arange(len(tiempo1))
bar_positions2 = np.arange(len(tiempo2)) + bar_width
plt.bar(bar_positions1, tiempo1, width=bar_width, label='GitHub', color='blue')
plt.bar(bar_positions2, tiempo2, width=bar_width, label='FedRec', color='red')
plt.xlabel('Índice do HVP')
plt.ylabel('Tiempo (s)')
plt.legend()

# Añadir un título común para ambas gráficas
plt.suptitle('Comparación das implementacións de L-BFGS', fontsize=16)

# Ajustar el diseño para evitar superposiciones
plt.tight_layout()

# Mostrar las gráficas
plt.savefig(ruta_base + '/comparacion.png')
plt.show()
