import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import math


class Client(object):
    def __init__(self, idx, model, train_fed_loader, val_fed_loader, test_fed_loader, bs, ep, lr, mask,
                 current_prune_ratio, threshold, prune_wait, device):
        self.idx = idx
        self.model = model
        self.bs = bs
        self.ep = ep
        self.lr = lr

        self.criterion = nn.MSELoss()
        self.train_fed_loader = DataLoader(train_fed_loader, batch_size=self.bs, shuffle=False)
        self.val_fed_loader = DataLoader(val_fed_loader, batch_size=self.bs, shuffle=False)
        self.test_fed_loader = DataLoader(test_fed_loader, batch_size=self.bs, shuffle=False)
        self.mask = mask
        self.current_prune_ratio = current_prune_ratio
        self.prune_wait = prune_wait
        self.device = device
        self.loss = 0
        self.threshold = threshold
        self.best_mask = mask
        self.best_model = model
        self.temp_model = None

    def get_mask(self):
        return self.mask

    def get_state_dict(self):
        return self.model.state_dict()

    def get_model(self):
        return self.model

    def set_state_dict(self, state_dict):
        return self.model.load_state_dict(state_dict)

    def process_loader(self, data):
        batch_weather = data[0:6]
        batch_soil = data[6:16]
        batch_management = data[16]
        batch_y = data[18]
        batch_pre_y = data[19]
        batch_pre_y = batch_pre_y.repeat_interleave(25, dim=1)          # boost
        batch_weather = torch.stack(batch_weather).permute(1, 0, 2, 3)
        batch_weather = batch_weather.view(batch_weather.size(0), batch_weather.size(1), -1)  # concat 5 years
        batch_soil = torch.stack(batch_soil).permute(1, 0, 2, 3)
        batch_soil = torch.mean(batch_soil, 2)  # only need 1 year since they are same
        batch_soil = batch_soil.view(batch_soil.size(0), -1)  # (6,10) into 60 since there are so few
        batch_management = batch_management.view(batch_management.size(0), -1)
        return batch_weather, batch_soil, batch_management, batch_y, batch_pre_y

    def train(self, model, freeze=False, early_stop=False):
        model.to(self.device)
        # curr_parameters = self.model.get_parameters()
        # self.model = model
        temp_model = model
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.0001)
        # optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        # scheduler = StepLR(optimizer, step_size=40, gamma=0.5)
        epoch_loss = []
        epoch_vloss = []
        tolerance = 10000
        count = 0

        if early_stop:
            tolerance = 2

        for epoch in range(self.ep):
            batch_loss = 0.0
            # current_lr = optimizer.param_groups[0]['lr']
            # print(current_lr)
            if count < tolerance:
                for step, data in enumerate(self.train_fed_loader):
                    batch_weather, batch_soil, batch_management, batch_y, batch_pre_y = self.process_loader(data)
                    batch_weather, batch_soil, batch_management, batch_y, batch_pre_y = \
                        batch_weather.to(self.device), \
                        batch_soil.to(self.device), \
                        batch_management.to(self.device), \
                        batch_y.to(self.device),    \
                        batch_pre_y.to(self.device)
                    optimizer.zero_grad()
                    prediction = model(batch_weather, batch_soil.unsqueeze(1), batch_management.unsqueeze(1), batch_pre_y)
                    loss = self.criterion(prediction, batch_y)
                    loss.backward()
                    if freeze:
                        # freeze pruned weights by making their gradients 0
                        for name, p in model.named_parameters():
                            if 'weight' in name:
                                tensor = p.data.cpu().numpy()
                                grad_tensor = p.grad.data.cpu().numpy()
                                grad_tensor = np.where(abs(tensor) < 0.000001, 0, grad_tensor)
                                p.grad.data = torch.from_numpy(grad_tensor).to(self.device)
                    optimizer.step()

                    batch_loss += loss.item()

                epoch_train_loss = math.sqrt(batch_loss / len(self.train_fed_loader))
                epoch_loss.append(epoch_train_loss)
                epoch_val_loss = self.eval_val(model)
                epoch_vloss.append(epoch_val_loss)
                print(f' Epoch {epoch + 1} loss: {epoch_train_loss} ' f'val_loss: {epoch_val_loss}')

                # curr_parameters = self.model.get_parameters()
                if early_stop and epoch > 0 and epoch_vloss[epoch-1] is not None:
                    if epoch_vloss[epoch] > epoch_vloss[epoch-1]:     # if current val loss > previous
                        count = count + 1
                        print('earlystopping count ', count)
                    else:
                        count = 0
        # scheduler.step()
        return model.state_dict(), epoch_loss

    def eval_val(self, model):
        # evaluation test
        val_loss = 0.0
        model.eval()
        for step, data in enumerate(self.val_fed_loader):
            batch_weather, batch_soil, batch_management, batch_y, batch_pre_y = self.process_loader(data)
            batch_weather, batch_soil, batch_management, batch_y, batch_pre_y = \
                batch_weather.to(self.device), \
                batch_soil.to(self.device), \
                batch_management.to(self.device), \
                batch_y.to(self.device), \
                batch_pre_y.to(self.device)
            val_pred = model(batch_weather, batch_soil.unsqueeze(1), batch_management.unsqueeze(1), batch_pre_y)
            loss = self.criterion(val_pred, batch_y)
            val_loss += loss.item()
        return math.sqrt(val_loss / len(self.val_fed_loader))