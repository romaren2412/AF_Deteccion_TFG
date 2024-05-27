import torch
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms

from MNIST.byzantine_models import backdoor, backdoor_sen_pixel, dba, edge, label_flip, mean_attack
from clases_redes import MnistNetFLARE, MnistNetFLTrust


class MNISTTraining:
    def __init__(self, c, trainloader, test_data=None) -> None:
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
        # self.net = MnistNetFLARE(num_channels=1, num_outputs=10).to(self.device)
        self.net = MnistNetFLTrust().to(self.device)

        self.dba_index = None
        self.byz = False
        self.trainloader = trainloader
        if test_data is not None:
            self.testloader = torch.utils.data.DataLoader(test_data, batch_size=500, shuffle=True,
                                                          generator=torch.Generator(device='cuda'))


class SupervisedLearning:
    def __init__(self, c, ap) -> None:
        self.c = c
        self.device = c.DEVICE
        self.ap = ap

    def adestrar(self, criterion, global_net, type_attack, target):
        grad_list = []
        rede = self.ap.net.to(self.device)
        rede.load_state_dict(global_net.state_dict())
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
            grad_list.append([param.grad.clone() for param in rede.parameters()])

        # Calcular la media de los gradientes
        avg_grads = []
        for grads in zip(*grad_list):
            avg_grads.append(sum(grads) / len(grads))

        return self.untargeted_attack(avg_grads, type_attack)

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
                return mean_attack(model)
        return model

    def adestrar_server(self, criterion, global_net):
        grad_list = []
        rede = self.ap.net.to(self.device)
        rede.load_state_dict(global_net.state_dict())
        optimizer = optim.SGD(rede.parameters(), lr=self.c.LR)
        for data in self.ap.trainloader:
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.type(torch.LongTensor).to(self.device)
            optimizer.zero_grad()
            outputs = rede(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            grad_list.append([param.grad.clone() for param in rede.parameters()])

        # Calcular la media de los gradientes
        avg_grads = []
        for grads in zip(*grad_list):
            avg_grads.append(sum(grads) / len(grads))
        return avg_grads

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


def inicializar_global_model(num_channels, num_outputs, device):
    # model = MnistNetFLARE(num_channels=num_channels, num_outputs=num_outputs)
    model = MnistNetFLTrust()
    return model.to(device)


def create_server_model(c, root_dataset):
    return MNISTTraining(c, root_dataset)


def create_local_models(num_clients, c, worker_loaders, test_data, byz_workers, global_net):
    aprendedores = []
    for i in range(num_clients):
        ap = MNISTTraining(c, worker_loaders[i], test_data)
        ap.net.load_state_dict(global_net.state_dict())
        if i in byz_workers:
            ap.byz = True
        aprendedores.append(ap)
    return aprendedores
