from graficas import plot_media_por_cliente, plot_evolucion_medias, debuxar_diferencias_precision, debuxar_precision


if __name__ == "__main__":
    gardar = True

    path = 'PROBAS/ProbasDef/mean_attack'
    save_path = path
    attack_type = path.split('/')[-1]

    # plot_media_por_cliente(path + '/score.csv')
    # plot_evolucion_medias(path + '/score.csv')
    debuxar_diferencias_precision('PROBAS_REF/FedAvg/no')
    # debuxar_precision(path + '/acc.csv')
