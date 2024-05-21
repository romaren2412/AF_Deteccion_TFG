from fldetector_mnistm import fl_detector
from DigitFive.D0_MNISTM_Ataques.config import parse_args, Config

if __name__ == "__main__":
    c = Config()
    args = parse_args()
    original_clients = [i for i in range(c.SIZE)]
    byz_workers = [i for i in range(args.nbyz)]
    n_entrenos = 0

    # IMPRIMIR INFORMACIÓN SOBRE EXECUCIÓN
    print("Parámetros de execución:")
    print("Número de épocas: ", c.EPOCH)
    print("Tipo de ataque: ", args.byz_type)
    print("Regra de agregación: ", args.aggregation)
    print("Home path: ", args.home_path)
    print("Tipo de execución: ", args.tipo_exec)
    print("Medida de confianza necesaria (silhouette): ", args.silhouette)

    if args.tipo_exec == 'detect':
        # TIPO DE EXECUCIÓN: DETECTAR (unha vez eliminado o cluster malicioso, detén o adestramento)
        detected_byz = fl_detector(c, args, original_clients, n_entrenos, original_clients, byz_workers)
        print("Clientes detectados como byzantinos: ", detected_byz)
        if detected_byz is None:
            real_clients = original_clients
        else:
            real_clients = [i for i in original_clients if i not in detected_byz]
        print("Clientes benignos: ", real_clients)

    elif args.tipo_exec == 'loop':
        real_clients = original_clients
        # TIPO DE EXECUCIÓN: BUCLE (continúa o adestramento eliminando os clientes maliciosos)
        detected_byz = fl_detector(c, args, original_clients, n_entrenos, original_clients, byz_workers)

        while detected_byz is not None:
            n_entrenos += 1
            real_clients = [i for i in real_clients if i not in detected_byz]
            print("----------------------------------")
            print("Iniciase novo entreno cos seguintes parámetros: ")
            print("Clientes detectados como byzantinos: ", detected_byz)
            print("Clientes benignos: ", real_clients)
            detected_byz = fl_detector(c, args, real_clients, n_entrenos, original_clients, byz_workers)

    elif args.tipo_exec == 'no_detect':
        # TIPO DE EXECUCIÓN: DETECTAR (unha vez eliminado o cluster malicioso, detén o adestramento)
        fl_detector(c, args, original_clients, n_entrenos, original_clients, byz_workers)

    else:
        raise NotImplementedError
