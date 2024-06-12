from flare_d5 import flare_d5
from config import Config

if __name__ == "__main__":
    c = Config()
    original_clients = [i for i in range(c.SIZE)]
    byz_workers = [i for i in range(c.NBYZ)]

    # IMPRIMIR INFORMACIÓN SOBRE EXECUCIÓN
    print("Parámetros de execución:")
    print("Número de épocas: ", c.EPOCH)
    print("Home path: ", c.home_path)

    flare_d5(c, original_clients, byz_workers)
