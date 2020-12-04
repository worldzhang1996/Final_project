import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from models import *
from experiment import experiment,quantization
from utils import init_distance,distance_evaluate,distance_train
import time
from torch.utils.data import TensorDataset,Dataset,DataLoader,random_split

exp = quantization()
PTQ = False
QAT = True

model = QDistModule()
model.load_state_dict(torch.load("/Users/zhangshijie/Desktop/COMP5933/experiment/models/distance_best_valid_model.pth",map_location = torch.device("cpu")))
print(model)

# prepare
model.to(exp.device)
model.eval()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_prepared = torch.quantization.prepare_qat(model, inplace=True)


# init train
# build data
distance_x = ['pickup_longitude_bin', 'pickup_latitude_bin', 'dropoff_longitude_bin', 'dropoff_latitude_bin',
                  'geodistance']
distance_y = 'trip_distance'

distance = pd.read_csv(exp.distance_path)
x = distance[distance_x].values
y = distance[distance_y].values.reshape(-1, 1)

print(x[0])
print(y[0])

distance_ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
print(type(distance_ds))
# 分割成训练集和预测集
n_train = int(len(distance) * 0.9)
n_valid = len(distance) - n_train

distance_ds_train, distance_ds_valid = random_split(distance_ds, [n_train, n_valid])
distance_loader_train, distance_loader_valid = DataLoader(distance_ds_train, batch_size=exp.batch_size), DataLoader(
        distance_ds_valid, batch_size=exp.batch_size)

# optimizer
optimizer = torch.optim.SGD(model_prepared.parameters(), lr=exp.lr, momentum=exp.momenta,
                                weight_decay=exp.weight_decay)

# loss function
criterion = torch.nn.MSELoss(reduce=True, size_average=True)

for epoch in range(10):
    model_prepared.train()
    for i, (inputs, labels) in enumerate(distance_loader_train):
        inputs, labels = inputs.to("cpu"), labels.to("cpu")
        inputs = inputs.float()
        outputs = model_prepared(inputs)[0]

        loss = criterion(outputs, labels.float())
        #print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 3000 == 0:
            valid_loss = distance_evaluate(model_prepared, criterion, distance_loader_valid, device="cpu")
            print("In epoch {},batches {} ,The valid loss is {}".format(epoch+1,i+1,valid_loss))
        if (i+1) > 50000:
            break
    if epoch > 1:
        # Freeze quantizer parameters
        model_prepared.apply(torch.quantization.disable_observer)

    quantized_model = torch.quantization.convert(model_prepared.eval(), inplace=False)
    valid_loss = distance_evaluate(quantized_model, criterion, distance_loader_valid, device="cpu")
    print("After epoch {},The valid loss is {}".format(epoch+1,valid_loss))