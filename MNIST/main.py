from MNIST.config import Config
from MNIST.flare import flare

if __name__ == "__main__":
    c = Config()
    original_clients = [i for i in range(c.SIZE)]
    byz_workers = [i for i in range(c.NBYZ)]

    # IMPRIMIR INFORMACIÓN SOBRE EXECUCIÓN
    print("Parámetros de execución:")
    print("Número de épocas: ", c.EPOCH)
    print("Tipo de ataque: ", c.byz_type)
    print("Home path: ", c.home_path)

    flare(c, original_clients, byz_workers)

