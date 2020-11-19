import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from models import *
from experiment import experiment,quantization
from utils import init_distance,distance_evaluate,distance_train
import time

exp = quantization()
PTQ = False
QAT = True
###################### DISTANCE #####################
# origin 2.4 compress 70

model = QDistModule()
model.load_state_dict(torch.load("/Users/zhangshijie/Desktop/COMP5933/experiment/models/distance_best_valid_model.pth",map_location = torch.device("cpu")))
print(model)

# evaluate
distance_model, optimizer, criterion, distance_loader_train, distance_loader_valid = init_distance(exp)
start = time.time() 
origin_loss = distance_evaluate(model, criterion, distance_loader_valid,device= exp.device)
print("Before quantization the time of model is {}".format(time.time()-start))
print("Before quantization the loss of model is {}".format(origin_loss))
torch.save(model.state_dict(), "origin.pth")
print('Size (MB):', os.path.getsize("origin.pth")/1e6)

# prepare
model.to(exp.device)
model.eval()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_prepared = torch.quantization.prepare_qat(model, inplace=True)

# training
#distance_train(exp, model_prepared, optimizer, criterion, distance_loader_train, distance_loader_valid,earlyStopping=None, device=exp.device)

# inference
model_int8 = torch.quantization.convert(model_prepared.eval())
    
start = time.time()
quantization_loss = distance_evaluate(model_int8, criterion, distance_loader_valid,device= exp.device)
print("After quantization the time of model is {}".format(time.time()-start))
print("After quantization the loss of model is {}".format(quantization_loss))
torch.save(model_int8, "QAT_compressed.pth")
print('Size (MB):', os.path.getsize("QAT_compressed.pth")/1e6)