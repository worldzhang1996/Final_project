import torch
from models import QDistModule
from utils import init_distance,init_time,time_train
from experiment import experiment,quantization

exp = experiment()
model = torch.load("QAT_compressed.pth")

time_model,optimizer,criterion,time_loader_train,time_loader_valid,x,y = init_time(exp,model)

time_train(exp, time_model, optimizer, criterion, time_loader_train, time_loader_valid)



