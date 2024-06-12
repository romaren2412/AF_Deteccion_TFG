import csv
import os

from graficas import plot_media_por_cliente, plot_evolucion_medias, debuxar_diferencias_precision, debuxar_precision


def engadir_encabezado(origen, destino, nuevo_header):
    with open(origen, mode='r', newline='', encoding='utf-8') as file_in:
        reader = csv.reader(file_in)

        with open(destino, mode='w', newline='', encoding='utf-8') as file_out:
            writer = csv.writer(file_out)
            writer.writerow(nuevo_header)  # Escribe el nuevo encabezado

            for row in reader:
                writer.writerow(row)  # Escribe todos los datos desde la primera línea


if __name__ == "__main__":
    gardar = True

    for data_maior in ['mnist', 'svhn', 'syn', 'usps', 'mnistm']:
        for data_menor in ['mnist', 'svhn', 'syn', 'usps', 'mnistm']:
            path1b = f'D5_1Batch/PROBAS_REF/{data_maior}/{data_maior}_{data_menor}'
            path_minibatches = f'D5/CBA/PROBAS/Script/{data_maior}_{data_menor}'
            path = path_minibatches
            if os.path.exists(path):
                save_path = path
                attack_type = f'#{data_maior}_{data_menor}'
                engadir_encabezado(path + '/score.csv', path + '/score2.csv',
                                   ['#0_Byz', '#1_Ben', '#2_Ben', '#3_Ben', '#4_Ben'])
                plot_media_por_cliente(path + '/score2.csv', attack_type, gardar=gardar, save_path=save_path)
                plot_evolucion_medias(path + '/score2.csv', attack_type, gardar=gardar, save_path=save_path)
                # debuxar_precision(path + '/acc.csv')
