import torch


def evaluate_accuracy(data_loader, model, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, label in data_loader:
            data, label = data.to(device).float(), label.to(device).float()
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    accuracy = correct / total
    return accuracy


def evaluate_backdoor(data_iterator, net, target, device, byz_type):
    net.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for i, (data, label) in enumerate(data_iterator):
            data, label = data.to(device), label.to(device)
            data[:, :, 26, 26] = 1
            data[:, :, 26, 24] = 1
            data[:, :, 25, 25] = 1
            data[:, :, 24, 26] = 1

            remaining_idx = list(range(data.shape[0]))
            for example_id in range(data.shape[0]):
                if label[example_id] == target:
                    remaining_idx.remove(example_id)
                else:
                    label[example_id] = target

            output = net(data)
            _, predictions = torch.max(output, 1)
            predictions = predictions[remaining_idx]
            label = label[remaining_idx]

            correct_predictions += (predictions == label).sum().item()
            total_samples += len(remaining_idx)

    accuracy = correct_predictions / total_samples
    return accuracy
