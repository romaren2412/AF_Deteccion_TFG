import torch
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms

from byzantine import backdoor, dba, label_flip, mean_attack, scaling_attack
from rede import MnistNet


class MNISTTraining:
    def __init__(self, c, trainloader, test_data=None) -> None:
        self.c = c
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.LR = c.LR
        self.BATCH_SIZE = c.BATCH_SIZE

        self.device = c.DEVICE
        self.epochs = c.EPOCH

        self.sl = SupervisedLearning(c, self)
        # self.net = MnistNetFLARE(num_channels=1, num_outputs=10).to(self.device)
        self.net = MnistNet().to(self.device)

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

    def adestrar(self, c, criterion, global_net, target):
        # Bucle de adestramento
        rede = self.ap.net.to(self.device)
        rede.load_state_dict(global_net.state_dict())
        optimizer = optim.SGD(rede.parameters(), lr=self.c.LR)

        local_epoch = 0
        while local_epoch < c.FL_FREQ:
            local_epoch += 1
            for data in self.ap.trainloader:
                inputs, labels = self.targeted_attack(data, c.byz_type, target)
                labels = labels.type(torch.LongTensor)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = rede(inputs.float())
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        self.untargeted_attack(c.byz_type)
        return self.calcular_diferencias(global_net), rede.state_dict()

    def targeted_attack(self, data, type_attack, target):
        if self.ap.byz:
            if type_attack == "backdoor":
                return backdoor(data, target)
            elif type_attack == "dba":
                return dba(data, target, self.ap.dba_index)
            elif type_attack == "label_flip":
                return label_flip(data)
        return data

    def untargeted_attack(self, type_attack):
        if self.ap.byz:
            if type_attack == "mean_attack":
                mean_attack(self.ap.net)
            if type_attack in ("backdoor", "dba"):
                scaling_attack(self.ap.net)

    def calcular_diferencias(self, global_net):
        local_params = self.ap.net.state_dict()
        global_params = global_net.state_dict()
        local_update = {key: local_params[key] - global_params[key] for key in global_params}
        return local_update

    def adestrar_server(self, c, criterion, global_net):
        rede = self.ap.net.to(self.device)
        rede.load_state_dict(global_net.state_dict())
        optimizer = optim.SGD(rede.parameters(), lr=self.c.LR)
        local_epoch = 0
        while local_epoch < c.FL_FREQ:
            local_epoch += 1
            for inputs, labels in self.ap.trainloader:
                inputs = inputs.to(self.device)
                labels = labels.type(torch.LongTensor).to(self.device)
                optimizer.zero_grad()
                outputs = rede(inputs.float())
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        return self.calcular_diferencias(global_net)

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


def inicializar_global_model(device):
    model = MnistNet()
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
