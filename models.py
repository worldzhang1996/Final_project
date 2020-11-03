import torch
import torch.nn as nn
import torch.nn.functional as F


class DistModule(nn.Module):
    def __init__(self):
        super(DistModule, self).__init__()
        self.layer1 = torch.nn.Linear(5, 32)
        self.layer2 = torch.nn.Linear(32, 64)
        self.layer3 = torch.nn.Linear(64, 16)
        self.layer4 = torch.nn.Linear(16, 1)
        # self.Sigmoid = nn.Sigmoid()
        # self.Relu = nn.ReLU6()

        # 初始化
        self._initialize_weights()

    def forward(self, x):
        x = F.sigmoid(self.layer1(x))
        # x = self.Relu(x)
        x = F.sigmoid(self.layer2(x))
        # x = self.Relu(x)
        x = F.sigmoid(self.layer3(x))
        # x = self.Relu(x)
        output = self.layer4(x)
        return output, x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class TimeModule(nn.Module):
    def __init__(self):
        super(TimeModule,self).__init__()
        self.layer1 = torch.nn.Linear(21,64)
        self.layer2 = torch.nn.Linear(64,128)
        self.layer3 = torch.nn.Linear(128,20)
        self.layer4 = torch.nn.Linear(20,1)
        self.Relu = torch.nn.ReLU6()
    
    def forward(self,x):
        x = self.layer1(x)
        x = self.Relu(x)
        x = self.layer2(x)
        x = self.Relu(x)
        x = self.layer3(x)
        x = self.Relu(x)
        return self.layer4(x)