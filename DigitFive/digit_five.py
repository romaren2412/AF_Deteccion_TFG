import torchvision.transforms as transforms
import numpy as np

import matplotlib.pyplot as plt
import torch
from DigitFive.methods import SupervisedLearning
import scipy.io


class DigitFiveDatasheet:
    def __init__(self, num, c, transform=None) -> None:
        self.id = {"mt": 0, "mm": 0, "sv": 0, "syn": 0, "us": 0}
        self.id_list = ["mt", "mm", "sv", "syn", "us"]
        self.last_data = np.array([59, 118, 191, 203, 213])
        self.images_root = []
        self.labels_root = []
        for i in num:
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
        print(f'[INFO USER {c.RANK}] numeros clientes {num[0]}/{num[-1]} datos {self.id}')

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


def show(img, label):
    plt.title('Label:' + str(label))
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


class Digit_Five_training():
    def __init__(self, c) -> None:
        """
        c-> configuration
        p-> porcentaxe de uso (probas dev, deberia ser o 1 para execucions reais)
        """
        self.c = c
        transform = transforms.Compose([transforms.ToTensor()])

        self.transform = transform
        indices = c.TRAIN_INDEX

        self.test_indices = np.array([])
        self.train_indices = list((set(indices)) - set(self.test_indices))

        self.LR = c.LR
        self.BATCH_SIZE = c.BATCH_SIZE
        self.BATCH_TEST_SIZE = c.BATCH_TEST_SIZE

        self.device = c.DEVICE
        self.epochs = c.EPOCH

        self.sl = SupervisedLearning(c, self)
        self.net = None
        self.dba_index = None
        self.byz = False

    def create_train(self):
        print(f'[INFO USER {self.c.RANK}] Creando trainloader')
        self.df_train = DigitFiveDatasheet(self.train_indices, self.c, self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.df_train, batch_size=self.BATCH_SIZE,
                                                       shuffle=True, num_workers=2,
                                                       generator=torch.Generator(device='cuda'))
        self.pre_data = next(iter(self.trainloader))

    def create_test(self):
        print(f'[INFO USER {self.c.RANK}] Creando testloader')
        self.df_test = DigitFiveDatasheet(self.c.TEST_INDEX_LOCAL, self.c, self.transform)
        self.testloader = torch.utils.data.DataLoader(self.df_test, batch_size=self.BATCH_TEST_SIZE,
                                                      shuffle=True, num_workers=2,
                                                      generator=torch.Generator(device='cuda'))
        self.data_test = next(iter(self.testloader))

    def create_test_global(self):
        self.df_test_global = DigitFiveDatasheet(self.c.TEST_INDEX_GLOBAL, self.c, self.transform)
        self.testloader_global = torch.utils.data.DataLoader(self.df_test_global, batch_size=self.BATCH_TEST_SIZE,
                                                             shuffle=True, num_workers=2,
                                                             generator=torch.Generator(device='cuda'))
        return self.testloader_global
