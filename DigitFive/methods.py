import numpy as np
import scipy.io
import torch
import torch.optim as optim
from torchvision import transforms

from redes import DigitFiveNet


class DigitFiveDatasheet:
    def __init__(self, num, c, transform=None) -> None:
        self.id = {"mt": 0, "mm": 0, "sv": 0, "syn": 0, "us": 0}
        self.id_list = ["mt", "mm", "sv", "syn", "us"]
        self.last_data = np.array([59, 118, 191, 203, 213])
        self.images_root = []
        self.labels_root = []
        for i in num:
            try:
                dict_mat = scipy.io.loadmat('./data/d5/' + 'data_' + str(i) + '.mat')
            except:
                dict_mat = scipy.io.loadmat('DigitFive/data/d5/' + 'data_' + str(i) + '.mat')
            l = dict_mat["labels"][0]  # .astype(float)
            im = dict_mat["images"]
            if len(self.labels_root) == 0:
                self.images_root = im
                self.labels_root = l
            else:
                self.images_root = np.concatenate((self.images_root, im))
                self.labels_root = np.concatenate((self.labels_root, l))
            data_base = np.argwhere(i <= self.last_data).flatten()[0]
            self.id[self.id_list[data_base]] += len(l)
        self.transform = transform
        print(f'[INFO USER {c.RANK}] Clientes: {num} || Datos: {self.id}')

    def __len__(self):
        return len(self.labels_root)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.images_root[idx]
        y = self.labels_root[idx]

        if self.transform:
            img = self.transform(img)
        sample = (img, y)

        return sample


class DigitFiveTraining:
    def __init__(self, c, indices, test_indices) -> None:
        """
        c-> configuration
        p-> porcentaxe de uso (probas dev, deberia ser o 1 para execucions reais)
        """
        self.c = c
        transform = transforms.Compose([transforms.ToTensor()])

        self.transform = transform

        self.test_indices = test_indices
        self.train_indices = indices

        self.LR = c.LR
        self.BATCH_SIZE = c.BATCH_SIZE
        self.BATCH_TEST_SIZE = c.BATCH_TEST_SIZE

        self.device = c.DEVICE
        self.epochs = c.EPOCH

        self.sl = SupervisedLearning(c, self)
        self.net = DigitFiveNet().to(self.device)
        self.dba_index = None
        self.byz = False

        self.create_train()
        self.create_test()

    def create_train(self):
        print(f'[INFO USER {self.c.RANK}] Creando trainloader')
        self.df_train = DigitFiveDatasheet(self.train_indices, self.c, self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.df_train, batch_size=self.BATCH_SIZE,
                                                       shuffle=True, num_workers=2,
                                                       generator=torch.Generator(device='cuda'))
        self.data_train = next(iter(self.trainloader))

    def create_test(self):
        print(f'[INFO USER {self.c.RANK}] Creando testloader')
        self.df_test = DigitFiveDatasheet(self.test_indices, self.c, self.transform)
        self.testloader = torch.utils.data.DataLoader(self.df_test, batch_size=self.BATCH_TEST_SIZE,
                                                      shuffle=True, num_workers=2,
                                                      generator=torch.Generator(device='cuda'))
        self.data_test = next(iter(self.testloader))

    def getCreateTest(self):
        return next(iter(self.testloader))


class SupervisedLearning:
    def __init__(self, c, ap) -> None:
        self.c = c
        self.device = c.DEVICE
        self.ap = ap

    def adestrar(self, criterion, global_net):
        rede = self.ap.net.to(self.device)
        rede.load_state_dict(global_net.state_dict())
        optimizer = optim.SGD(rede.parameters(), lr=self.c.LR)

        inputs, labels = self.ap.data_train
        # inputs, labels = self.ap.pre_data
        labels = labels.type(torch.LongTensor)
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        optimizer.zero_grad()

        outputs = rede(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        return self.calcular_diferencias(global_net), rede.state_dict()

    def calcular_diferencias(self, global_net):
        local_params = self.ap.net.state_dict()
        global_params = global_net.state_dict()
        local_update = {key: local_params[key] - global_params[key] for key in global_params}
        return local_update

    def test(self, net, testloader):
        net = net.to(self.device)
        correct = 0
        total = 0
        with torch.no_grad():
            # for data in testloader:
            images, labels = self.ap.data_test
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = net(images.float())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return correct / total
