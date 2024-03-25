import csv
from evaluate import *


def gardar_puntuacions(mal_score, path):
    with open(path + '/score.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(mal_score)


def gardar_precisions(path, precision_array, attack_type):
    with open(path + '/acc.csv', 'w', newline='') as csvFile:
        csvwriter = csv.writer(csvFile)
        if attack_type in ('backdoor', 'dba', 'backdoor_sen_pixel', 'edge'):
            csvwriter.writerow(["Iteracions", "ACC_Global", "ASR"])
        else:
            csvwriter.writerow(["Iteracions", "ACC_Global"])
        csvwriter.writerows(precision_array)


def testear_precisions(testloader_global, global_net, device, e, precision_array, path, target, test_edge, byz_type):
    acc_global = evaluate_accuracy(testloader_global, global_net, device)
    if byz_type in ('backdoor', 'dba', 'backdoor_sen_pixel'):
        backdoor_sr = evaluate_backdoor(testloader_global, global_net, target=target, device=device, type=byz_type)
        print("Epoch %02d. Train_acc %0.4f Attack_sr %0.4f" % (e, acc_global, backdoor_sr))
        precision_array.append([e, acc_global, backdoor_sr])
    elif byz_type == 'edge':
        backdoor_edge_sr = evaluate_edge_backdoor(test_edge, global_net, device)
        print("Epoch %02d. Train_acc %0.4f Attack_sr %0.4f" % (e, acc_global, backdoor_edge_sr))
        precision_array.append([e, acc_global, backdoor_edge_sr])
    else:
        print("Epoch %02d. Train_acc %0.4f" % (e, acc_global))
        precision_array.append([e, acc_global])
    gardar_precisions(path, precision_array, byz_type)


def resumo_final(test_data_loader, net, device, e, malicious_score, path):
    test_accuracy = evaluate_accuracy(test_data_loader, net, device)
    print("Epoch %02d. Test_acc %0.4f" % (e, test_accuracy))
    gardar_puntuacions(malicious_score, path)


def datos_finais(file_path, precision_array, test_data_loader, net, device, e, malicious_score, byz_type):
    gardar_precisions(file_path, precision_array, byz_type)
    # PRECISION FINAL + Malicious score
    resumo_final(test_data_loader, net, device, e, malicious_score, file_path)
