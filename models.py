import torch
import torch.nn as nn
class DistModule(nn.Module):
    def __init__(self):
        super(DistModule,self).__init__()
        self.layer1 = torch.nn.Linear(4,20)
        self.layer2 = torch.nn.Linear(20,100)
        self.layer3 = torch.nn.Linear(100,20)
        self.layer4 = torch.nn.Linear(20,1)
        self.Relu = nn.ReLU6()
    
    def forward(self,x):
        x = self.layer1(x)
        x = self.Relu(x)
        x = self.layer2(x)
        x = self.Relu(x)
        x = self.layer3(x)
        x = self.Relu(x)
        output = self.layer4(x)
        return output,x

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