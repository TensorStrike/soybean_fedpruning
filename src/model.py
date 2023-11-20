import torch
import torch.nn as nn
from torch.nn.utils.prune import l1_unstructured
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader


class SoybeanCNN(nn.Module):
    def __init__(self):
        super(SoybeanCNN, self).__init__()
        # weather input:6x260(5 years)
        self.w_conv1 = nn.Conv1d(in_channels=6, out_channels=8, kernel_size=9, stride=1,
                               padding='valid')                                                            # (260-9)/1+1=252
        self.w_bn1 = nn.BatchNorm1d(8)
        self.w_pool1 = nn.AvgPool1d(kernel_size=2, stride=2)                                                # 126
        self.w_conv2 = nn.Conv1d(in_channels=8, out_channels=12, kernel_size=3, stride=1,
                               padding='valid')                                                            # (126-3)/1+1=124
        self.w_bn2 = nn.BatchNorm1d(12)
        self.w_pool2 = nn.AvgPool1d(kernel_size=2, stride=2)                                               # 62
        self.w_conv3 = nn.Conv1d(in_channels=12, out_channels=16, kernel_size=3, stride=1,
                               padding='valid')                                                            # (62-3)/1+1=60
        self.w_bn3 = nn.BatchNorm1d(16)
        self.w_pool3 = nn.AvgPool1d(kernel_size=2, stride=2)                                               # 30
        self.w_conv4 = nn.Conv1d(in_channels=16, out_channels=20, kernel_size=3, stride=1,                  # (30-3)/1+1=28
                               padding='valid')
        self.w_bn4 = nn.BatchNorm1d(20)
        self.w_pool4 = nn.MaxPool1d(kernel_size=2, stride=2)                                                # 14
        # self.w_fc1 = nn.Linear(in_features=20*14, out_features=40)

        # soil input: 60
        self.s_conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding='valid')       # (60-3)/1+1 = 58
        self.s_bn1 = nn.BatchNorm1d(4)
        self.s_pool1 = nn.AvgPool1d(kernel_size=2, stride=2)                                                    # 29
        self.s_conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding='valid')       # (29-3)/1+1 = 27
        self.s_bn2 = nn.BatchNorm1d(8)
        self.s_pool2 = nn.AvgPool1d(kernel_size=2, stride=2)                                                    # 13
        self.s_conv3 = nn.Conv1d(in_channels=8, out_channels=12, kernel_size=2, stride=1, padding='valid')      # (13-2)/1+1 = 12
        self.s_bn3 = nn.BatchNorm1d(12)
        self.s_pool3 = nn.AvgPool1d(kernel_size=2, stride=2)                                                    # 6
        # self.s_fc1 = nn.Linear(in_features=12*6, out_features=40)

        # management: 70
        self.m_conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding='valid')       # (70-3)/1+1 = 68
        self.m_bn1 = nn.BatchNorm1d(4)
        self.m_pool1 = nn.AvgPool1d(kernel_size=2, stride=2)                                                    # 34
        self.m_conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding='valid')       # (34-3)/1+1 = 32
        self.m_bn2 = nn.BatchNorm1d(8)
        self.m_pool2 = nn.AvgPool1d(kernel_size=2, stride=2)                                                    # 16
        # self.m_fc1 = nn.Linear(in_features=8*16, out_features=20)

        # concatenated
        self.c_bn1 = nn.BatchNorm1d(580)         # concat w+s+m+y
        self.c_fc1 = nn.Linear(in_features=580, out_features=240)
        self.c_bn2 = nn.BatchNorm1d(240)
        self.c_fc2 = nn.Linear(in_features=240, out_features=60)
        self.c_bn3 = nn.BatchNorm1d(60)
        self.c_fc3 = nn.Linear(in_features=60, out_features=1)
        # self.c_bn4 = nn.BatchNorm1d(32)
        # self.c_fc4 = nn.Linear(in_features=32, out_features=1)
        #
        # layer_list = []
        # layer_list.append(self.w_conv1)
        # layer_list.append(self.w_conv2)
        # layer_list.append(self.w_conv3)
        # layer_list.append(self.w_conv4)
        # layer_list.append(self.w_fc1)
        # layer_list.append(self.s_conv1)
        # layer_list.append(self.s_conv2)
        # layer_list.append(self.s_conv3)
        # layer_list.append(self.s_fc1)
        # layer_list.append(self.m_conv1)
        # layer_list.append(self.m_conv2)
        # layer_list.append(self.m_fc1)
        # layer_list.append(self.c_fc1)
        # layer_list.append(self.c_fc2)
        # layer_list.append(self.c_fc3)
        #
        # self.module_list = nn.ModuleList(layer_list)
        #
        # self.track_layers = {'w_conv1': self.w_conv1, 'w_conv2': self.w_conv2, 'w_conv3': self.w_conv3,
        #                      'w_conv4': self.w_conv4, 'w_fc1': self.w_fc1,
        #                      's_conv1': self.s_conv1, 's_conv2': self.s_conv2, 's_conv3': self.s_conv3,
        #                      's_fc1': self.s_fc1,
        #                      'm_conv1': self.m_conv1, 'm_conv2': self.m_conv2, 'm_fc1': self.m_fc1,
        #                      'c_fc1': self.c_fc1, 'c_fc2': self.c_fc2, 'c_fc3': self.c_fc3}
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_uniform_(module.weight.data, nonlinearity='relu', mode='fan_in')
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight.data, mode='fan_in')
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)

    def forward(self, x1, x2, x3, x4):
        x1 = self.w_pool1(self.w_bn1(F.relu(self.w_conv1(x1))))
        x1 = self.w_pool2(self.w_bn2(F.relu(self.w_conv2(x1))))
        x1 = self.w_pool3(self.w_bn3(F.relu(self.w_conv3(x1))))
        x1 = self.w_pool4(self.w_bn4(F.relu(self.w_conv4(x1))))
        # print(x.size())
        x1 = x1.view(x1.size(0), -1)  # flatten   (batch, 20*n)
        # x1 = F.relu(self.w_fc1(x1))

        x2 = self.s_pool1(self.s_bn1(F.relu(self.s_conv1(x2))))
        x2 = self.s_pool2(self.s_bn2(F.relu(self.s_conv2(x2))))
        x2 = self.s_pool3(self.s_bn3(F.relu(self.s_conv3(x2))))
        x2 = x2.view(x2.size(0), -1)
        # x2 = F.relu(self.s_fc1(x2))

        x3 = self.m_pool1(self.m_bn1(F.relu(self.m_conv1(x3))))
        x3 = self.m_pool2(self.m_bn2(F.relu(self.m_conv2(x3))))
        x3 = x3.view(x3.size(0), -1)
        # x3 = F.relu(self.m_fc1(x3))

        c = torch.cat((x1, x2, x3, x4), dim=1)
        c = self.c_bn1(F.relu(c))
        output = self.c_bn2(F.relu(self.c_fc1(c)))
        output = self.c_bn3(F.relu(self.c_fc2(output)))
        output = self.c_fc3(output)

        '''
        x1 = self.w_pool1(F.relu(self.w_conv1(x1)))
        x1 = self.w_pool2(F.relu(self.w_conv2(x1)))
        x1 = self.w_pool3(F.relu(self.w_conv3(x1)))
        x1 = self.w_pool4(F.relu(self.w_conv4(x1)))
        # print(x.size())
        x1 = x1.view(x1.size(0), -1)  # flatten   (batch, 20*n)
        x1 = F.relu(self.w_fc1(x1))
        # x1 = self.w_fc2(x1)

        x2 = self.s_pool1(F.relu(self.s_conv1(x2)))
        x2 = self.s_pool2(F.relu(self.s_conv2(x2)))
        x2 = self.s_pool3(F.relu(self.s_conv3(x2)))
        x2 = x2.view(x2.size(0), -1)
        x2 = F.relu(self.s_fc1(x2))
        # x2 = self.s_fc2(x2)

        x3 = self.m_pool1(F.relu(self.m_conv1(x3)))
        x3 = self.m_pool2(F.relu(self.m_conv2(x3)))
        x3 = x3.view(x3.size(0), -1)
        x3 = F.relu(self.m_fc1(x3))

        c = torch.cat((x1, x2, x3), dim=1)
        c = F.relu(c)
        output = F.relu(self.c_fc1(c))
        output = F.relu(self.c_fc2(output))
        output = self.c_fc3(output)
        '''
        return output

    def get_parameters(self):
        parameters_dict = dict()
        for layer_name in self.track_layers:
            parameters_dict[layer_name] = {
                'weight': self.track_layers[layer_name].weight.data,
                'bias': self.track_layers[layer_name].bias.data
            }
        return parameters_dict

    def apply_parameters(self, parameters_dict):
        with torch.no_grad():
            for layer_name in parameters_dict:
                self.track_layers[layer_name].weight.data *= 0
                self.track_layers[layer_name].bias.data *= 0
                self.track_layers[layer_name].weight.data += parameters_dict[layer_name]['weight']
                self.track_layers[layer_name].bias.data += parameters_dict[layer_name]['bias']

    # def evaluate(self, data_loader, bs, device):
    #     data_loader = DataLoader(data_loader, batch_size=bs, shuffle=False)
    #     losses = []
    #     self.eval()
    #     for step, data in enumerate(data_loader):
    #         batch_weather = data[0:6]
    #         batch_soil = data[6:16]
    #         batch_management = data[16]
    #         batch_y = data[18]
    #         batch_weather = torch.stack(batch_weather).permute(1, 0, 2, 3)
    #         batch_weather = batch_weather.view(batch_weather.size(0), batch_weather.size(1), -1)  # concat 5 years
    #         batch_soil = torch.stack(batch_soil).permute(1, 0, 2, 3)
    #         batch_soil = torch.mean(batch_soil, 2)  # only need 1 year since they are same
    #         batch_soil = batch_soil.view(batch_soil.size(0), -1)  # (6,10) into 60 since there are so few
    #         batch_management = batch_management.view(batch_management.size(0), -1)
    #
    #         batch_weather, batch_soil, batch_management, batch_y = \
    #             batch_weather.to(device), \
    #             batch_soil.to(device), \
    #             batch_management.to(device), \
    #             batch_y.to(device)
    #
    #         val_pred = self(batch_weather, batch_soil.unsqueeze(1), batch_management.unsqueeze(1))
    #         criterion = nn.MSELoss()
    #         loss = criterion(val_pred, batch_y)
    #         losses.append(loss)
    #     return math.sqrt(torch.stack(losses).mean().item())
    #