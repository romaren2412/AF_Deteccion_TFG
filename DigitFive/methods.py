import torch


class SupervisedLearning:
    def __init__(self, c, ap) -> None:
        self.c = c
        self.device = c.DEVICE
        self.ap = ap

    def adestrar(self, criterion, rede, grad_list):
        data = self.ap.pre_data
        inputs, labels = data
        labels = labels.type(torch.LongTensor)
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        rede.zero_grad()

        outputs = rede(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        grad_list.append([param.grad.clone() for param in rede.parameters()])

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
