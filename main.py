import torch
from experiment import experiment
from utils import *


def main(exp):
    torch.set_default_tensor_type(torch.FloatTensor)

    # distance
    distance_model,optimizer,criterion,distance_loader_train,distance_loader_valid = init_distance(exp)
    distance_train(exp,distance_model,optimizer,criterion,distance_loader_train,distance_loader_valid,device = "cpu")
    
    # time
    time_model,optimizer,criterion,time_loader_train,time_loader_valid = init_time(exp,distance_model)
    time_train(exp,time_model,optimizer,criterion,time_loader_train,time_loader_valid)


if __name__=="__main__":
    random_seed()
    exp = experiment()
    main(exp)