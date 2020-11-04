import torch

from EarlyStopping import EarlyStopping
from experiment import experiment
from utils import *


def main(exp):
    torch.set_default_tensor_type(torch.FloatTensor)

    # distance
    distance_model, optimizer, criterion, distance_loader_train, distance_loader_valid = init_distance(exp)
    distance_train(exp,distance_model,optimizer,criterion,distance_loader_train,distance_loader_valid,EarlyStopping(patience=3, mode='min'))
    # distance_model.load_state_dict(torch.load('./models/distance_20201104072611.pth', map_location=torch.device('cpu')))
    
    # time
    time_model, optimizer, criterion, time_loader_train, time_loader_valid, x, y = init_time(exp, distance_model)
    time_train(exp, time_model, optimizer, criterion, time_loader_train, time_loader_valid, earlyStopping=EarlyStopping(patience=3, mode='min'))

import torch

if __name__=="__main__":
    print("模型文件是1.6版本下的 当前版本为: ",torch.__version__)
    random_seed()
    exp = experiment(distance_path="./data/trip_data_1_normalized_distance_trip.csv",time_path="./data/trip_data_1_normalized_time_trip.csv")
    path = './data/trip_data_1.csv'
    gen_distance_dataset(path, normalize=True)
    gen_time_dataset(path, normalize_target=True)
    main(exp)