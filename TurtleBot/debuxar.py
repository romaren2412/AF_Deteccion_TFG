from graficas import plot_media_por_cliente, plot_evolucion_medias, debuxar_precision

if __name__ == "__main__":
    gardar = True

    for data_maior in ["1", "2", "izq", "der"]:
        for data_menor in ["1", "2", "izq", "der"]:
            path = f'PROBAS/TurtleBotScript2/4_{data_maior}/{data_menor}'
            save_path = path
            attack_type = f'#{data_maior}_{data_menor}'
            plot_media_por_cliente(path + '/score.csv', attack_type, gardar, save_path)
            plot_evolucion_medias(path + '/score.csv', attack_type, gardar, save_path)
            # debuxar_precision(path + '/acc.csv')
