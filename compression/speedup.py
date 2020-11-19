import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
import nni
import argparse
from models import DistModule,TimeModule
import datetime
import torch
from nni.compression.torch import ModelSpeedup,apply_compression_results

def speedup(exp):

    # load the model
    
    if exp.model=="distance":
        model = DistModule()
        torch.save(model,exp.save_path+"/origin.pth")
        #model.load_state_dict(torch.load(exp.model_path,map_location=exp.device))
    elif exp.model=="time":
        model = TimeModule()
        #model.load_state_dict(torch.load(exp.model_path,map_location=exp.device))
    else:
        raise Exception("This model {} has not been supported!".format(exp.model))
    

    # dummy_input 
    dummy_input = torch.randn(64,5)

    apply_compression_results(model, args.mask_path,args.device)
    m_speedup = ModelSpeedup(model, dummy_input.to(exp.device), args.mask_path)
    m_speedup.speedup_model()
    
    # save model
    save_path = exp.save_path+"/"+str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))+"_"+exp.model+"_speedup.pth"
    torch.save(model,save_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser("The speedup part")
    parser.add_argument("--model",type = str)
    parser.add_argument("--model_path",type = str)
    parser.add_argument("--mask_path",type = str)
    parser.add_argument("--save_path",type = str)
    parser.add_argument("--device",type = str,default = "cpu")
    args = parser.parse_args()
    speedup(args)