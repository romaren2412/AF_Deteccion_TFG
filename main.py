from config import parse_args
from mnist import adestrar

if __name__ == "__main__":
    args = parse_args()
    original_clients = [i for i in range(args.nworkers)]
    byz_workers = [i for i in range(args.nbyz)]

    # IMPRIMIR INFORMACIÓN SOBRE EXECUCIÓN
    print("Parámetros de execución:")
    print("Número de épocas: ", args.nepochs)
    print("Tipo de ataque: ", args.byz_type)
    print("Regra de agregación: ", args.aggregation)
    print("Home path: ", args.home_path)
    print("-----------------------------------")
    print("Número de clientes: ", args.nworkers)
    print("Benignos: ", [i for i in original_clients if i not in byz_workers])
    print("Byzantinos: ", byz_workers)

    detected_byz = adestrar(args, original_clients)
