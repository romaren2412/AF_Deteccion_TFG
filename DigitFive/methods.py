import torch
from DigitFive.byzantine import backdoor, backdoor_sen_pixel, dba, edge, label_flip


class SupervisedLearning:
    def __init__(self, c, ap) -> None:
        self.c = c
        self.device = c.DEVICE
        self.ap = ap

    def adestrar(self, criterion, rede, grad_list, type_attack, target):
        data = self.ap.pre_data
        inputs, labels = self.targeted_attack(data, type_attack, target)
        labels = labels.type(torch.LongTensor)
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        rede.zero_grad()

        outputs = rede(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        grad_list.append([param.grad.clone() for param in rede.parameters()])

    def targeted_attack(self, data, type_attack, target):
        if self.ap.byz:
            if type_attack == "backdoor":
                return backdoor(data, target)
            elif type_attack == "backdoor_sen_pixel":
                return backdoor_sen_pixel(data, target)
            elif type_attack == "dba":
                return dba(data, target, self.ap.dba_index)
            elif type_attack == "edge":
                return edge(data, target)
            elif type_attack == "label_flip":
                return label_flip(data)
        return data

    def test(self, net, testloader):
        # since we're not training, we don't need to calculate the gradients for our outputs
        net = net.to(self.device)
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data

                images = images.to(self.device)
                labels = labels.to(self.device)
                # calculate outputs by running images through the network
                outputs = net(images.float(), False)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
        return correct / total
