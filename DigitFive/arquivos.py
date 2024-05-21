# coding: utf-8
import csv
from evaluate import *


def gardar_precisions(path, precision_array, tipo):
    if tipo == 'cinco_diferentes':
        row = ["Iteracions", "ACC_MNIST", "ACC_MNISTM", "ACC_SVHN", "ACC_SYN", "ACC_USPS", "ACC_Global"]
    else:
        row = ["Iteracions", "ACC_Base", "ACC_Extra"]
    with open(path + '/acc.csv', 'w', newline='') as csvFile:
        csvwriter = csv.writer(csvFile)
        csvwriter.writerow(row)
        csvwriter.writerows(precision_array)


def testear_precisions(aprendedores, testloader_global, global_net, device, e, precision_array, path, data_type,
                       extra_data_type, tipo):
    print(f"---\n[EPOCH: {e}] - PROBAS DE PRECISIÓN\n---")
    if tipo == 'cinco_diferentes':
        acc_mnist = acc_mnistm = acc_svhn = acc_syn = acc_usps = 0
        for i, ap in enumerate(aprendedores):
            if i == 0:
                acc_mnist = evaluate_accuracy(ap.testloader, global_net, device)
                print(f"[ACC MNIST #{i}] " + str(acc_mnist))
            if i == 1:
                acc_mnistm = evaluate_accuracy(ap.testloader, global_net, device)
                print(f"[ACC MNISTM #{i}] " + str(acc_mnistm))
            if i == 2:
                acc_svhn = evaluate_accuracy(ap.testloader, global_net, device)
                print(f"[ACC SVHN #{i}] " + str(acc_svhn))
            if i == 3:
                acc_syn = evaluate_accuracy(ap.testloader, global_net, device)
                print(f"[ACC SYN #{i}] " + str(acc_syn))
            if i == 4:
                acc_usps = evaluate_accuracy(ap.testloader, global_net, device)
                print(f"[ACC USPS #{i}] " + str(acc_usps))
        precisions = [e, acc_mnist, acc_mnistm, acc_svhn, acc_syn, acc_usps]
    else:
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
    gardar_precisions(path, precision_array, tipo)


def gardar_puntuacions(mal_score, path):
    with open(path + '/score.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(mal_score)


def resumo_final(test_data_loader, net, device, e, malicious_score, path):
    test_accuracy = evaluate_accuracy(test_data_loader, net, device)
    print("Epoch %02d. Test_acc %0.4f" % (e, test_accuracy))
    gardar_puntuacions(malicious_score, path)


def datos_finais(file_path, precision_array, test_data_loader, net, device, e, malicious_score, attack_type):
    gardar_precisions(file_path, precision_array, attack_type)
    # PRECISION FINAL + Malicious score
    resumo_final(test_data_loader, net, device, e, malicious_score, file_path)
