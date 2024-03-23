from config import *
from MNIST_Simple import fl_detector

if __name__ == "__main__":
    args = parse_args()
    original_clients = [i for i in range(args.nworkers)]
    n_entrenos = 0

    # IMPRIMIR INFORMACIÓN SOBRE EXECUCIÓN
    print("Parámetros de execución:")
    print("Número de épocas: ", args.nepochs)
    print("Tipo de ataque: ", args.byz_type)
    print("Regra de agregación: ", args.aggregation)
    print("Home path: ", args.home_path)
    print("Tipo de execución: ", args.tipo_exec)
    print("Medida de confianza necesaria (silhouette): ", args.silhouette)

    detected_byz = fl_detector(args, original_clients, n_entrenos, original_clients)