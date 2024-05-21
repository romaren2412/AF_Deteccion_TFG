import torch


class SupervisedLearning:
    def __init__(self, c, ap) -> None:
        self.c = c
        self.device = c.DEVICE
        self.ap = ap

    def adestrar_tb(self, criterion, grad_list):
        rede = self.ap.net.to(self.device)
        trainloader = self.ap.trainloader

        data = next(iter(trainloader))
        inputs, labels, _ = data
        labels = labels.type(torch.FloatTensor)
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        rede.zero_grad()

        outputs = rede(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        grad_list.append([param.grad.clone() for param in rede.parameters()])


class Helpers:
    def save_data(self, nombre, lista, red=False, how="w", bd=False):
        """ Para gardar os datos de adestramento """
        if red:
            torch.save(lista, nombre)
        else:
            f = open(nombre, how)
            for li in lista:
                f.write((str(li) + " ")) if not bd else f.write((' '.join(map(str, li))) + '\n')
            f.write("\n")
            f.close()

    def show_wheigh(self, net, label="", show_parts=False):
        print(label + "first layer", net.state_dict()[next(iter(net.state_dict()))][0][0])
        if show_parts:
            print(label + "shared", net.a.state_dict()[next(iter(net.a.state_dict()))][0][0])
            print(label + "local", net.b.state_dict()[next(iter(net.b.state_dict()))][0])
