import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def test_tb(testloader, net, criterion, device):
    # since we're not training, we don't need to calculate the gradients for our outputs
    net = net.to(device)
    loss = 0.0

    dd_list = []
    ld_list = []
    with torch.no_grad():
        for data in testloader:
            images, labels, _ = data

            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images.float())
            # the class with the highest energy is what we choose as prediction
            loss += criterion(outputs, labels).item()
            dd, ld = test_tb_discret(outputs.cpu().numpy(), labels.cpu().numpy())
            if len(dd_list) == 0:
                dd_list = dd
                ld_list = ld
            else:
                dd_list = np.concatenate((dd_list, dd))
                ld_list = np.concatenate((ld_list, ld))
    cm = confusion_matrix(ld_list, dd_list, labels=[0, 1, 2, 3])
    accuracy = calculate_accuracy_tb(dd_list, ld_list)
    return cm.flatten(), loss, accuracy


def calculate_accuracy_tb(predicted_labels, actual_labels):
    correct_predictions = (predicted_labels == actual_labels).sum().item()
    total_predictions = len(actual_labels)
    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy


def test_tb_discret(data, labels):
    """
    0- Queto
    1- Adiamte
    2- Dereita
    3- Izquierda
    """
    dd = np.zeros(data.shape[0])
    ld = np.zeros(labels.shape[0])

    # data[:,0] #velocidades lineares
    # data[:,1] #velocidades angulares

    gir_izq = np.argwhere(data[:, 1] > 0.3).flatten()
    gir_der = np.argwhere(data[:, 1] < -0.3).flatten()
    ad = np.argwhere(data[:, 0] > 0.1).flatten()

    dd[ad] = 1
    dd[gir_izq] = 2
    dd[gir_der] = 3

    gir_izq = np.argwhere(labels[:, 1] > 0.3)
    gir_der = np.argwhere(labels[:, 1] < -0.3)
    ad = np.argwhere(labels[:, 0] > 0.1)

    ld[ad] = 1
    ld[gir_izq] = 2
    ld[gir_der] = 3

    return dd, ld
