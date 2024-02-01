import matplotlib.pyplot as plt

def grafica(data, iter, type):
    # Dibujar el ndarray en un gráfico
    plt.plot(data, 'o')
    plt.xlabel('Índice do cliente')
    plt.ylabel('Malicious score')
    plt.title('CNN: Iteración ' + str(iter) + ' - ' + type)
    plt.show()

def grafica_gardar_byz(data, iter, type, path, undetected_byz_index):
    colores = ['red' if i in undetected_byz_index else 'blue' for i in range(len(data))]
    # Dibujar el ndarray en un gráfico
    plt.scatter(range(len(data)), data, color=colores)
    plt.xlabel('Índice do cliente')
    plt.ylabel('Malicious score')
    plt.title('Iteración ' + str(iter) + ' - ' + type)

    # Crear manualmente dos objetos para la leyenda
    clientes_bizantinos = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10)
    clientes_benignos = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10)

    # Añadir leyenda
    plt.legend([clientes_bizantinos, clientes_benignos], ['Clientes byzantinos', 'Clientes benignos'])
    plt.savefig(path + '/Iter_' + str(iter) + '.png')


def grafica_gardar(data, iter, type, path):
    # Dibujar el ndarray en un gráfico
    plt.plot(data, 'o')
    plt.xlabel('Índice do cliente')
    plt.ylabel('Malicious score')
    plt.title('CNN: Iteración ' + str(iter) + ' - ' + type)
    plt.savefig(path + '/CNN_Iter_' + str(iter) + '_' + type + '.png')

def grafica_gardar2(data, iter, type, path):
    # Dibujar el ndarray en un gráfico
    plt.plot(data, 'o')
    plt.xlabel('Índice do cliente')
    plt.ylabel('Malicious score')
    plt.title('CNN: Iteración ' + str(iter) + ' - ' + type)
    plt.savefig(path + '/' + type + '_' + str(iter) + '.png')
