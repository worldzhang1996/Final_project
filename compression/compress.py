import nni
import os
from compress_utils import *

"""
There, we try to build a pipeline for compression. 
Its contains: pruning, distallization and quantization.
More specifically, 
1. pruning：searching best trail,speedup, distallization+finetuning.
2. quantization: XXX.

"""
# hyperparameter control
class setting:
    def __init__(self):
        self.save_path = ""
        self.device = "cpu"

def main(exp):
    # modify nni yaml

    # run nni
    cmd = "nnictl create --config nni/examples/trials/mnist-tfv1/config.yml"
    os.system(cmd)

    # wait trail
    while(1):
        pass

    # speedup
    speedup()

    # finetuning
    finetuning()

    # save the model


if __name__=="__main__":
    exp = setting()
    main(exp)