import argparse
import datetime
from mnist_CNN import fl_detector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset", default='mnist', type=str)
    parser.add_argument("--bias", help="degree of non-IID to assign data to workers", type=float, default=0.1)
    # parser.add_argument("--net", help="net", default='mlr', type=str, choices=['mlr', 'cnn', 'fcnn'])
    parser.add_argument("--batch_size", help="batch size", default=32, type=int)
    parser.add_argument("--lr", help="learning rate", default=0.0002, type=float)
    parser.add_argument("--nworkers", help="# workers", default=100, type=int)
    parser.add_argument("--nepochs", help="# epochs", default=500, type=int)
    parser.add_argument("--gpu", help="index of gpu", default=0, type=int)
    parser.add_argument("--seed", help="seed", default=41, type=int)
    parser.add_argument("--nbyz", help="# byzantines", default=28, type=int)
    parser.add_argument("--byz_type", help="type of attack", default='no', type=str,
                        choices=['no', 'partial_trim', 'full_trim', 'mean_attack', 'full_mean_attack', 'gaussian',
                                 'dir_partial_krum_lambda', 'dir_full_krum_lambda', 'label_flip', 'backdoor', 'dba',
                                 'edge'])
    parser.add_argument("--aggregation", help="aggregation rule", default='simple_mean', type=str,
                        choices=['simple_mean', 'trim', 'krum', 'median'])

    # Engadidas por Roi
    parser.add_argument("--home_path", help="home path", default='', type=str)
    parser.add_argument("--tipo_exec", help="tipo de execución", default='detect', type=str,
                        choices=['detect', 'loop'])
    parser.add_argument("--silhouette", help="medida de confianza necesaria (silhouette)", default=0.75,
                        type=float)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    original_clients = [i for i in range(args.nworkers)]
    n_entrenos = 0

    if args.tipo_exec == 'detect':
        # TIPO DE EXECUCIÓN: DETECTAR (unha vez eliminado o cluster malicioso, detén o adestramento)
        detected_byz = fl_detector(args, original_clients, n_entrenos, original_clients)
        print("Clientes detectados como byzantinos: ", detected_byz)
        if detected_byz is None:
            real_clients = original_clients
        else:
            real_clients = [i for i in original_clients if i not in detected_byz]
        print("Clientes benignos: ", real_clients)

    elif args.tipo_exec == 'loop':
        real_clients = original_clients
        # TIPO DE EXECUCIÓN: BUCLE (continúa o adestramento eliminando os clientes maliciosos)
        detected_byz = fl_detector(args, original_clients, n_entrenos, original_clients)

        while detected_byz is not None:
            n_entrenos += 1
            real_clients = [i for i in real_clients if i not in detected_byz]
            print("----------------------------------")
            print("Iniciase novo entreno cos seguintes parámetros: ")
            print("Clientes detectados como byzantinos: ", detected_byz)
            print("Clientes benignos: ", real_clients)
            detected_byz = fl_detector(args, real_clients, n_entrenos, original_clients)

    else:
        raise NotImplementedError
