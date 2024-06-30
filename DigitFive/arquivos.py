# coding: utf-8
import csv
from evaluate import *


def gardar_precisions(path, precision_array):
    row = ["Iteracions", "ACC_Base", "ACC_Extra"]
    with open(path + '/acc.csv', 'w', newline='') as csvFile:
        csvwriter = csv.writer(csvFile)
        csvwriter.writerow(row)
        csvwriter.writerows(precision_array)


def testear_precisions(aprendedores, testloader_global, global_net, device, e, precision_array, path, data_type,
                       extra_data_type):
    print(f"---\n[EPOCH: {e}] - PROBAS DE PRECISIÓN\n---")
    acc_mnistm = acc_extra = 0
    for i, ap in enumerate(aprendedores):
        if i != 4:
            acc_mnistm += evaluate_accuracy(ap.testloader, global_net, device)
        else:
            acc_extra = evaluate_accuracy(ap.testloader, global_net, device)
            print(f"[ACC {extra_data_type}] " + str(acc_extra))
    acc_mnistm = acc_mnistm / 4
    print(f"[ACC {data_type}] " + str(acc_mnistm))
    precisions = [e, acc_mnistm, acc_extra]
    acc_global = evaluate_accuracy(testloader_global, global_net, device)
    print(f"[Net Global]" + str(acc_global))
    precision_array.append(precisions)
    gardar_precisions(path, precision_array)


def gardar_puntuacions(mal_score, path, byz_index):
    headers = []
    for i in range(len(mal_score[0])):
        client_type = 'Byz' if i in byz_index else 'Ben'
        headers.append(f'#{i}_{client_type}')
    with open(path + '/score.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(mal_score)


def resumo_final(test_data_loader, net, device, e, malicious_score, path, byz_index):
    test_accuracy = evaluate_accuracy(test_data_loader, net, device)
    print("Epoch %02d. Test_acc %0.4f" % (e, test_accuracy))
    gardar_puntuacions(malicious_score, path, byz_index)


def datos_finais(file_path, precision_array, test_data_loader, net, device, e, malicious_score, attack_type, byz_index):
    gardar_precisions(file_path, precision_array, attack_type)
    # PRECISION FINAL + Malicious score
    resumo_final(test_data_loader, net, device, e, malicious_score, file_path, byz_index)
