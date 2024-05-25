import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data
import torch.nn as nn
from copy import deepcopy

from MNIST.byzantine import backdoor, backdoor_sen_pixel, dba, edge, label_flip, mean_attack_v2
from clases_redes import MnistNetFLARE


class MNISTTraining:
    def __init__(self, c, trainloader, test_data) -> None:
        """
        c-> configuration
        p-> porcentaxe de uso (probas dev, deberia ser o 1 para execucions reais)
        """
        self.c = c
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.LR = c.LR
        self.BATCH_SIZE = c.BATCH_SIZE

        self.device = c.DEVICE
        self.epochs = c.EPOCH

        self.sl = SupervisedLearning(c, self)
        self.net = MnistNetFLARE(num_channels=1, num_outputs=10).to(self.device)

        self.dba_index = None
        self.byz = False
        self.trainloader = trainloader
        self.testloader = torch.utils.data.DataLoader(test_data, batch_size=500, shuffle=True,
                                                      generator=torch.Generator(device='cuda'))


class SupervisedLearning:
    def __init__(self, c, ap) -> None:
        self.c = c
        self.device = c.DEVICE
        self.ap = ap

    def adestrar(self, criterion, type_attack, target):
        rede = self.ap.net.to(self.device)
        optimizer = optim.SGD(rede.parameters(), lr=self.c.LR)

        for data in self.ap.trainloader:
            inputs, labels = self.targeted_attack(data, type_attack, target)
            labels = labels.type(torch.LongTensor)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            outputs = rede(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        sent_model = self.untargeted_attack(deepcopy(rede), type_attack)
        return sent_model.state_dict()

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

    def untargeted_attack(self, model, type_attack):
        if self.ap.byz:
            if type_attack == "mean_attack":
                return mean_attack_v2(model)
        return model

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
                outputs = net(images.float())
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
        acc = correct / total
        return acc


def inicializar_global_model(num_channels, num_outputs, device, aux_loader, lr):
    model = MnistNetFLARE(num_channels=num_channels, num_outputs=num_outputs)
    model.to(device)
    return pretrain_global_model(model, aux_loader, device, lr)


def pretrain_global_model(model, data_loader, device, lr, epochs=5):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    return model


def create_local_models(num_clients, c, worker_loaders, test_data, byz_workers, global_net):
    aprendedores = []
    for i in range(num_clients):
        ap = MNISTTraining(c, worker_loaders[i], test_data)
        ap.net.load_state_dict(global_net.state_dict())
        if i in byz_workers:
            ap.byz = True
        aprendedores.append(ap)
    return aprendedores
