import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
import nni
import argparse
from nni.compression.torch.pruning import LevelPruner
from models import DistModule,TimeModule
import datetime
import torch

def pruner(exp):

    path = exp.save_path
    if exp.model=="distance":
        model = DistModule()
        #model.load_state_dict(torch.load(exp.model_path,map_location=exp.device))
    elif exp.model=="time":
        model = TimeModule()
        #model.load_state_dict(torch.load(exp.model_path,map_location=exp.device))
    else:
        raise Exception("This model {} has not been supported!".format(exp.model))
    model.to(exp.device)
    print(model)
    


if __name__=="__main__":
    parser = argparse.ArgumentParser("The pruner")
    parser.add_argument("--model",type = str)
    parser.add_argument("--model_path",type = str)
    parser.add_argument("--save_path",type = str)
    parser.add_argument("--device",type = str,default = "cpu")
    args = parser.parse_args()
    pruner(args)
