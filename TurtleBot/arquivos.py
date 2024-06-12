import csv
from evaluate_tb import *


def gardar_puntuacions(score, path, byzantine_index):
    headers = []
    for i in range(len(score[0])):
        client_type = 'Byz' if i in byzantine_index else 'Ben'
        headers.append(f'#{i}_{client_type}')

    with open(path + '/score.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(headers)
        writer.writerows(score)


def gardar_precisions(precision_path, precision_array):
    with open(precision_path + '/acc.csv', 'w', newline='') as csvFile:
        csvwriter = csv.writer(csvFile)
        csvwriter.writerow(["Iteracions", "Precision"])
        csvwriter.writerows(precision_array)


def testear_precisions(testloader_global, global_net, device, e, precision_array, path, criterion):
    cm, loss, acc_global = test_tb(testloader_global, global_net, criterion, device)
    print(f"Epoch {e} Train_acc {acc_global: .2f}% Loss {loss}, CM {cm}")
    precision_array.append([e, acc_global])
    gardar_precisions(path, precision_array)


def resumo_final(test_data_loader, net, device, e, malicious_score, path, criterion):
    _, _, test_accuracy = test_tb(test_data_loader, net, criterion, device)
    print("Epoch %02d. Test_acc %0.2f" % (e, test_accuracy))
    gardar_puntuacions(malicious_score, path, [0])


def gardar_precisions_locais(path, precision_array, byzantine_index):
    headers = []
    for i in range(len(precision_array[0])):
        client_type = 'Byz' if i in byzantine_index else 'Ben'
        headers.append(f'#{i}_{client_type}')

    with open(path + '/local_acc.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(headers)
        writer.writerows(precision_array)
