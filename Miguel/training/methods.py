import torch
from clases_redes import Net_Miguel as Net
from byzantine import backdoor, backdoor_sin_pixel, dba, edge, label_flip


class supervised_learning():
    def __init__(self, c, ap) -> None:
        self.c = c
        self.device = c.DEVICE
        self.trust_crit_baye = []
        self.ap = ap

    def adestrar(self, criterion, grad_list, type_attack, target, tipo):
        if tipo == 'fora':
            self.adestrar_fora(criterion, grad_list, type_attack, target)
        elif tipo == 'dentro':
            self.adestrar_dentro(criterion, grad_list, type_attack, target)
        elif tipo == 'sen_step':
            self.adestrar_sen_step(criterion, grad_list, type_attack, target)

    def adestrar_sen_step(self, criterion, grad_list, type_attack, target):
        rede = self.ap.net.to(self.device)
        trainloader = self.ap.trainloader
        optimizer = self.ap.optimizer
        grad_acumulados = []
        for step, data in enumerate(trainloader, 0):
            inputs, labels = self.targeted_attack(data, type_attack, target)
            labels = labels.type(torch.LongTensor)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = rede(inputs.float())
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()

            if step == 0:
                grad_acumulados = [param.grad for param in rede.parameters()]
            else:
                # Acumulamos los gradientes
                for i, param in enumerate(rede.parameters()):
                    grad_acumulados[i] += param.grad

        grad_list.append([grad.clone() for grad in grad_acumulados])

    def adestrar_fora(self, criterion, grad_list, type_attack, target):
        rede = self.ap.net.to(self.device)
        trainloader = self.ap.trainloader
        optimizer = self.ap.optimizer
        grad_acumulados = []
        optimizer.zero_grad()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = self.targeted_attack(data, type_attack, target)
            labels = labels.type(torch.LongTensor)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = rede(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            grad_acumulados.append([param.grad.clone() for param in rede.parameters()])
            optimizer.step()
        grad_acumulados = [torch.sum(torch.stack(grads), dim=0) for grads in zip(*grad_acumulados)]
        grad_list.append([grad.clone() for grad in grad_acumulados])

    def adestrar_dentro(self, criterion, grad_list, type_attack, target):
        rede = self.ap.net.to(self.device)
        trainloader = self.ap.trainloader
        optimizer = self.ap.optimizer
        grad_acumulados = []
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            inputs, labels = self.targeted_attack(data, type_attack, target)
            labels = labels.type(torch.LongTensor)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = rede(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            grad_acumulados.append([param.grad.clone() for param in rede.parameters()])
            optimizer.step()
        grad_acumulados = [torch.mean(torch.stack(grads), dim=0) for grads in zip(*grad_acumulados)]
        grad_list.append([grad.clone() for grad in grad_acumulados])

    def targeted_attack(self, data, type_attack, target):
        if self.ap.byz:
            if type_attack == "backdoor":
                return backdoor(data, target)
            elif type_attack == "backdoor_sin_pixel":
                return backdoor_sin_pixel(data, target)
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
