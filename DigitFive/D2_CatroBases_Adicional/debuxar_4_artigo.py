#coding: utf-8
import os
import pandas as pd
from graficas import *

import numpy as np


def debuxar_medias_artigo(opt, prepared_data):
    FLDET_START = 50

    sliced_data = prepared_data[::10]

    # Adjust the iteration numbers with FLDET_START and then filter up to iteration 1000
    iterations = sliced_data[:, 0] + FLDET_START
    mask = iterations <= 1000
    filtered_data = sliced_data[mask]
    iterations = iterations[mask]

    # Separate Byzantine and benign data from the filtered data
    # O BIZANTINO É O PRIMEIRO
    data_byz = filtered_data[:, 1]
    data_benigna = np.mean(filtered_data[:, 2:], axis=1)  # Mean of the rest of the columns

    # Continue with plotting...
    # Plot Byzantine data
    plt.plot(iterations, data_byz, 'r-', label=f'Suspicious score of the different client #{opt["data_byz"]}')

    # Plot benign data
    plt.plot(iterations, data_benigna, 'g--', label=f'Average suspicious score of common clients #{opt["data_ben"]}')

    plt.xlabel('Training epoch')
    # plt.title(f'Evolución dos scores - {opt["data_ben"]} vs {opt["data_byz"]}')
    # plt.legend(loc='upper left')
    # Adjust the y-axis to leave space for the legend at the bottom
    y_min, y_max = plt.ylim()
    plt.ylim(y_min - 0.05, y_max)  # Extend the y-axis lower bound by 0.1
    plt.grid(True)
    # Move the legend to the bottom left
    plt.legend(loc='lower left')

    plt.show()


def prepare_data_with_iterations(data, start):
    """
    Prepares the data by adding an iteration column at the beginning.

    :param data: A NumPy array of shape (n_rows, n_columns).
    :return: A new NumPy array with an additional iteration column.
    """
    n_rows, n_columns = data.shape
    iteration_column = np.arange(start, n_rows + start).reshape(n_rows, 1)  # Create an iteration column
    prepared_data = np.hstack((iteration_column, data))  # Concatenate the iteration column to the original data

    return prepared_data


if __name__ == "__main__":
    path = 'PROBAS/V2/mnist_syn/0'
    save_path = os.path.dirname(path)
    data_benigna = 'MNISTM'
    data_maliciosa = 'syn'

    opt = {"gardar": True,
           "detectar": True,
           "save_path": save_path,
           "tipo": "catro_bases",
           "data_ben": data_benigna,
           "data_byz": data_maliciosa}

    data_score_artigo = pd.read_csv(path + '/score.csv', header=None)
    data_score_prepared = prepare_data_with_iterations(data_score_artigo, start=50)
    debuxar_medias_artigo(opt, data_score_prepared)
