from DigitFive.D5_1Batch.fltrust_d5_1b import fltrust_mnist
from DigitFive.config import Config

if __name__ == "__main__":
    c = Config('1b')
    original_clients = [i for i in range(c.SIZE)]
    byz_workers = [i for i in range(c.NBYZ)]

    # IMPRIMIR INFORMACIÓN SOBRE EXECUCIÓN
    print("Parámetros de execución:")
    print("Número de épocas: ", c.EPOCH)
    print("Home path: ", c.home_path)

    fltrust_mnist(c, original_clients, byz_workers)
