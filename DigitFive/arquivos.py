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
    acc_common = acc_extra = 0
    for i, ap in enumerate(aprendedores):
        if i != 0:
            acc_common += evaluate_accuracy(ap.testloader, global_net, device)
        else:
            acc_extra = evaluate_accuracy(ap.testloader, global_net, device)
            print(f"[ACC {extra_data_type}] " + str(acc_extra))
    acc_common = acc_common / 4
    print(f"[ACC {data_type}] " + str(acc_common))
    precisions = [e, acc_common, acc_extra]
    acc_global = evaluate_accuracy(testloader_global, global_net, device)
    print(f"[Net Global]" + str(acc_global))
    precision_array.append(precisions)
    gardar_precisions(path, precision_array)


def testear_precisions_un_batch(aprendedores, testloader_global, global_net, device, e, precision_array, path, data_type,
                       extra_data_type):
    print(f"---\n[EPOCH: {e}] - PROBAS DE PRECISIÓN\n---")
    acc_common = acc_extra = 0
    for i, ap in enumerate(aprendedores):
        if i != 0:
            acc_common += evaluate_accuracy_un_batch(ap.data_test, global_net, device)
        else:
            acc_extra = evaluate_accuracy_un_batch(ap.data_test, global_net, device)
            print(f"[ACC {extra_data_type}] " + str(acc_extra))
    acc_common = acc_common / 4
    print(f"[ACC {data_type}] " + str(acc_common))
    precisions = [e, acc_common, acc_extra]
    acc_global = evaluate_accuracy_un_batch(testloader_global, global_net, device)
    print(f"[Net Global]" + str(acc_global))
    precision_array.append(precisions)
    gardar_precisions(path, precision_array)


def gardar_puntuacions(mal_score, path):
    with open(path + '/score.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(mal_score)


def resumo_final(test_data_loader, net, device):
    test_accuracy = evaluate_accuracy(test_data_loader, net, device)
    print("Epoca final. Test_acc %0.4f" % test_accuracy)


def resumo_final_un_batch(test_data_loader, net, device):
    test_accuracy = evaluate_accuracy_un_batch(test_data_loader, net, device)
    print("Epoca final. Test_acc %0.4f" % test_accuracy)
