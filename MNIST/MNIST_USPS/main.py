from MNIST.config import parse_args
from MNIST.MNIST_USPS.fld_mnist_usps import fld_mnist_usps

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

    if args.tipo_exec == 'detect':
        # TIPO DE EXECUCIÓN: DETECTAR (unha vez eliminado o cluster malicioso, detén o adestramento)
        detected_byz = fld_mnist_usps(args, original_clients, n_entrenos, original_clients)
        print("Clientes detectados como byzantinos: ", detected_byz)
        if detected_byz is None:
            real_clients = original_clients
        else:
            real_clients = [i for i in original_clients if i not in detected_byz]
        print("Clientes benignos: ", real_clients)

    elif args.tipo_exec == 'no_detect':
        # TIPO DE EXECUCIÓN: DETECTAR (unha vez eliminado o cluster malicioso, detén o adestramento)
        detected_byz = fld_mnist_usps(args, original_clients, n_entrenos, original_clients)

    else:
        raise NotImplementedError
