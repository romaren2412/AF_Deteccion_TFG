from fltrust_tb import fltrust_tb
from config import Config

if __name__ == "__main__":
    c = Config()
    original_clients = [i for i in range(c.SIZE)]

    # IMPRIMIR INFORMACIÓN SOBRE EXECUCIÓN
    print("Parámetros de execución:")
    print("Número de épocas: ", c.EPOCH_tb)
    print("Home path: ", c.home_path)

    fltrust_tb(c, original_clients)
