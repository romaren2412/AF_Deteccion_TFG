import os
import pandas as pd

import Miguel.config
from Miguel.config import parse_args
from graficas import debuxar_medias, grafica_cluster_ataque, leer_undetected, debuxar_precision_ataque


args = parse_args()
FLDET_START = Miguel.config.Config().FLDET_START


if __name__ == "__main__":
    path = 'PROBAS/D/backdoor/median/0'
    save_path = os.path.dirname(path)
    tipo_ataque = path.split('/')[-3]

    opt = {"gardar": True,
           "detectar": True,
           "save_path": save_path,
           "tipo": "ataque",
           "ataque": tipo_ataque}

    data_score = pd.read_csv(path + '/score.csv', header=None).T.values.tolist()
    data_acc = pd.read_csv(path + '/acc.csv', header=0)
    undetected = leer_undetected(path + '/Ataques_Detectados.txt')

    debuxar_medias(opt, data_score, undetected)
    grafica_cluster_ataque(opt, data_score, undetected)
    debuxar_precision_ataque(opt, data_acc)
