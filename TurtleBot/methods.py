import torch
import torch.optim as optim


class SupervisedLearning:
    def __init__(self, c, ap) -> None:
        self.c = c
        self.device = c.DEVICE
        self.ap = ap

    def adestrar_tb(self, criterion, global_net):
        rede = self.ap.net.to(self.device)
        rede.load_state_dict(global_net.state_dict())
        optimizer = optim.SGD(rede.parameters(), lr=self.c.LR_tb)

        local_epoch = 0
        while local_epoch < self.c.FL_FREQ:
            local_epoch += 1
            for data in self.ap.trainloader:
                # data = next(iter(self.ap.trainloader))
                inputs, labels, _ = data
                labels = labels.type(torch.FloatTensor)
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
