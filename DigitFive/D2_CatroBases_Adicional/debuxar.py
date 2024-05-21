import os
import pandas as pd

import Miguel.config
from Miguel.config import parse_args
from graficas import debuxar_medias, debuxar_precision_catro_bases, leer_undetected

args = parse_args()
FLDET_START = Miguel.config.Config().FLDET_START

if __name__ == "__main__":
    path = 'PROBAS/Net_Miguel/MNIST_Resto/mnist_syn/0'
    save_path = os.path.dirname(path)
    data_benigna = 'MNISTM'
    data_maliciosa = 'syn'

    opt = {"gardar": True,
           "detectar": True,
           "save_path": save_path,
           "tipo": "catro_bases",
           "data_ben": data_benigna,
           "data_byz": data_maliciosa}

    data_score = pd.read_csv(path + '/score.csv', header=None).T.values.tolist()
    data_acc = pd.read_csv(path + '/acc.csv', header=0)
    undetected = leer_undetected(path + '/Ataques_Detectados.txt')

    debuxar_medias(opt, data_score, undetected)
    debuxar_precision_catro_bases(opt, data_acc, data_benigna, data_maliciosa)
