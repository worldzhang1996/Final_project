# 运行文件命令行 FLASK_ENV=development FLASK_APP=serving.py flask run

import sys
sys.path.append("/Users/zhangshijie/Desktop/COMP5933/experiment/")

import torch
from flask import Flask,jsonify
import io
import torch
from flask import request
from models import QDistModule,TimeModule,DistModule
import json

class setting:
    def __init__(self):
        self.distance_path = "/Users/zhangshijie/Desktop/COMP5933/experiment/models/distance_best_valid_model.pth"
        self.time_path = "/Users/zhangshijie/Desktop/COMP5933/experiment/models/time_20201104112215.pth"


exp = setting()
# load the model
distance = DistModule()
distance.load_state_dict(torch.load(exp.distance_path,map_location = torch.device("cpu")))
time = TimeModule()
time.load_state_dict(torch.load(exp.time_path,map_location = torch.device("cpu")))

def preprocess(data):
    return data
def prepare_data(data):
    # json
    pickup_longitude = data["pickup_longitude"]
    pickup_latitude = data["pickup_latitude"]
    dropoff_longitude = data["dropoff_longitude"] 
    dropoff_latitude = data["dropoff_latitude"]
    pickup_datetime = data["pickup_datetime"]
    geodistance = data['geodistance']
    distance_data,time_data = preprocess([pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,geodistance]),pickup_datetime
    distance_data,time_data = torch.tensor(distance_data),torch.tensor([time_data])
    return distance_data,time_data

def get_prediction(data):

    distance_data,time_data = prepare_data(data)
    print(distance_data,time_data)
    distance_out = distance(distance_data)
    print(type(distance_out),type(time_data))
    time_data = torch.cat((distance_out[1],time_data),0)
    time_out = time(time_data)
    return time_out.item()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':
        # we will get the file from the request
        file = request.files['file']
        # convert that to bytes
        data = file.read()
        data = json.loads(data)
        ETA = get_prediction(data)

        return jsonify({'ETA': ETA})


if __name__ == '__main__':
    app.run()