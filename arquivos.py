import csv
from evaluate import evaluate_accuracy


def gardar_puntuacions(mal_score, path):
    with open(path + '/score.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(mal_score)


def gardar_precisions(path, precision_array):
    with open(path + '/acc.csv', 'w', newline='') as csvFile:
        csvwriter = csv.writer(csvFile)
        csvwriter.writerow(["Iteracions", "ACC_Global"])
        csvwriter.writerows(precision_array)


def testear_precisions(testloader_global, global_net, device, e, precision_array, path):
    acc_global = evaluate_accuracy(testloader_global, global_net, device)
    precision_array.append([e, acc_global])
    gardar_precisions(path, precision_array)
    return acc_global


def resumo_final(test_data_loader, net, device, e, malicious_score, path):
    test_accuracy = evaluate_accuracy(test_data_loader, net, device)
    print("Epoch %02d. Test_acc %0.4f" % (e, test_accuracy))
    gardar_puntuacions(malicious_score, path)


def datos_finais(file_path, precision_array, test_data_loader, net, device, e, malicious_score):
    gardar_precisions(file_path, precision_array)
    # PRECISION FINAL + Malicious score
    resumo_final(test_data_loader, net, device, e, malicious_score, file_path)
