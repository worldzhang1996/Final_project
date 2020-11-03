import pandas as pd
import torch
from torch.utils.data import TensorDataset,Dataset,DataLoader,random_split
from experiment import experiment
from models import DistModule,TimeModule
from datetime import datetime
import numpy as np
import random
import os
from sklearn.preprocessing import MinMaxScaler
from torch.optim import SGD, lr_scheduler, Adam

from math import radians, cos, sin, asin, sqrt
def geodistance(lng1,lat1,lng2,lat2):
    if lng1 == lng2 and lat1 == lat2: return 0.0
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    # 经纬度转换成弧度
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371 # 地球平均半径，6371km
    # distance=round(distance,0)
    return distance  #返回m

distance_x = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude','dropoff_latitude']
distance_y = 'trip_distance'

def process_lon_lat(df):
    lon, lat = -74.0000, 40.43
    range = 0.5
    df_processed = df.loc[((lon-range)<=df.pickup_longitude)&((lon+range)>=df.pickup_longitude)&((lat-range/3)<=df.pickup_latitude)&((lat+range)>=df.pickup_latitude)]
    return df_processed

def drop_na(df,cols):
    for col in cols:
        df = df.loc[~df[col].isna()]
    return df
def distance_mapping(df, cluster_num = 60000):
    mm_list = []
    cols = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude','dropoff_latitude']
    for col in cols:
        print(f"************mapping {col}************")
        mm = MinMaxScaler()
        df[f'{col}_bin'] = mm.fit_transform(df[col].values.reshape(-1, 1)).reshape(-1)
        df[f'{col}_bin'] = (df[f'{col}_bin']*cluster_num).astype(int)
        df[f'{col}_bin'] = (df[f'{col}_bin'].astype(float))/cluster_num
        mm_list.append(mm)
    return df, mm_list


def gen_distance_dataset(path, normalize=None):
    base_path = path[:-4]
    data = pd.read_csv(path)
    # 增加直线经纬度信息
    data['geodistance'] = data.apply(
        lambda row: geodistance(row['pickup_longitude'], row['pickup_latitude'], row['dropoff_longitude'],
                                row['dropoff_latitude']), axis=1)
    data = process_lon_lat(data)
    # deal with distance
    names = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "geodistance",
             "trip_distance"]
    distance = data[names]

    distance = drop_na(distance, ['dropoff_longitude'])
    distance, mm_list = distance_mapping(distance)

    if normalize:
        final_path = base_path + "_normalized" + "_distance_trip.csv"
        # names = ["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude","geodistance"]
        names = ['pickup_longitude_bin', 'pickup_latitude_bin', 'dropoff_longitude_bin', 'dropoff_latitude_bin',
                 'geodistance']
        for name in names:
            mean = np.nanmean(distance[name].values)
            std = np.nanstd(distance[name].values)
            distance[name] = (distance[name] - mean) / std

    else:
        final_path = base_path + "_distance_trip.csv"
    print("distance_dataset： ", final_path)
    distance.to_csv(final_path, index=False)


    
def gen_time_dataset(path,normalize = None):
    base_path = path[:-4]
    data = pd.read_csv(path)
    # deal with Time
    names = ["pickup_datetime","dropoff_datetime","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"]
    times = data[names]
    times["trip_period"] = pd.to_datetime(times["dropoff_datetime"])-pd.to_datetime(times["pickup_datetime"])
    times["trip_period"] = times["trip_period"].dt.total_seconds()
    del times["dropoff_datetime"]
    
    min_date = pd.to_datetime(times["pickup_datetime"]).min()
    
    times["pickup_datetime"] = pd.to_datetime(times["pickup_datetime"])-min_date
    times["pickup_datetime"] = times["pickup_datetime"].dt.total_seconds()
    
    x_max = times["pickup_datetime"].max()
    x_min = times["pickup_datetime"].min()
    times["pickup_datetime"] = (times["pickup_datetime"]-x_min)/(x_max-x_min)
    
    if normalize:
        final_path = base_path+"_normalized"+"_time_trip.csv"
        names = ["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude","trip_period"]
        for name in names:
            if name =="trip_period":
                mean = times[name].mean()
                std = times[name].std()
                times[name] = (times[name]-mean)/(std)
                print(x_max,x_min)
                print(times)
    else:
        final_path = base_path+"_time_trip.csv"
    times.to_csv(final_path,index = False)

def prepare_data(path): 
    gen_distance_dataset(path,normalize = True)
    gen_time_dataset(path,normalize = True)

def random_seed(seed = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

##### distance ###########
def init_distance(exp):
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

    # model
    distance_model = DistModule()
    # 参数初始化
    # distance_model.apply(weights_init)

    # optimizer
    optimizer = torch.optim.SGD(distance_model.parameters(), lr=exp.lr, momentum=exp.momenta,
                                weight_decay=exp.weight_decay)
    # OPTIM=Adam(distance_model.parameters(), lr=exp.lr, weight_decay=exp.weight_decay)

    # loss function
    criterion = torch.nn.MSELoss(reduce=True, size_average=True)

    return distance_model, optimizer, criterion, distance_loader_train, distance_loader_valid


def distance_train(exp, distance_model, optimizer, criterion, distance_loader_train, distance_loader_valid,
                   earlyStopping=None, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    print("正在使用:", device)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    columns = ["epoch", "samples", "train_loss", "valid_loss"]
    log = pd.DataFrame(columns=columns)

    distance_model.to(device)
    distance_model.train()

    reduceLR = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)

    print("len(distance_loader_train) is ", len(distance_loader_train))

    exit_flag = False
    for epoch in range(exp.epochs):
        if exit_flag:
            break
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(distance_loader_train):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.float()
            outputs = distance_model(inputs)[0]

            loss = criterion(outputs, labels.float())
            # print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            batchs_numm2print = 3000
            if (i + 1) % batchs_numm2print == 0:
                now_loss = running_loss / batchs_numm2print
                valid_loss = distance_evaluate(distance_model, criterion, distance_loader_valid, device=device)
                # print("In epoch {} and {} samples, train loss is {}, valid loss is {} ".format((epoch+1),(i+1),now_loss,valid_loss))
                log = pd.DataFrame(
                    {"epoch": [epoch + 1], "samples": [i + 1], "train_loss": [now_loss], "valid_loss": [valid_loss]},
                    columns=["epoch", "samples", "train_loss", "valid_loss"])
                # lr衰减设置
                reduceLR.step(now_loss)
                path = "./log/distance_" + timestamp + ".csv"

                if i + 1 == batchs_numm2print:
                    log.to_csv(path, mode="a", index=False, header=True)
                else:
                    log.to_csv(path, mode="a", index=False, header=False)

                # 早停设置
                if (i + 1) % 10000 == 0:
                    print("In epoch {} and {} samples, train loss is {}, valid loss is {} ".format((epoch + 1), (i + 1),
                                                                                                   now_loss,
                                                                                                   valid_loss))
                    if earlyStopping is not None:
                        if earlyStopping.step(valid_loss):
                            path = "./models/distance_" + datetime.now().strftime('%Y%m%d%H%M%S_earlystop') + ".pth"
                            torch.save(distance_model.state_dict(), path)
                            print("earlyStopping.... model file already in ", path)
                            exit_flag = True
                            break

                running_loss = 0.0

    # save the model
    # path = "./models/distance_"+timestamp+".pth"
    # torch.save(distance_model.state_dict(),path)
    print("训练完成")
    


def distance_evaluate(distance_model,criterion,distance_loader_valid,device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    distance_model.eval()
    total,total_loss = len(distance_loader_valid),0
    for i,(inputs,labels) in enumerate(distance_loader_valid):
        inputs,labels = inputs.to(device),labels.to(device)
        inputs = inputs.float()

        outputs = distance_model(inputs)[0]
        loss = criterion(outputs,labels.float())
        if not np.isnan(loss.item()):
            total_loss += loss.item()/total
            #print(total_loss,loss.item(),total)
    distance_model.train()
    return total_loss

###### time ########
def init_time(exp,distance_model):
    distance = pd.read_csv(exp.distance_path)
    time = pd.read_csv(exp.time_path)
    distance_inputs = distance[distance_x].values
    distance_labels = distance[distance_y].values.reshape(-1,1)

    distance_ds = TensorDataset(torch.from_numpy(distance_inputs),torch.from_numpy(distance_labels))
    distance_dataloader = DataLoader(distance_ds,batch_size=256)
    distance_outputs = np.array([[0]*20])
    for (inputs,labels) in distance_dataloader:
        inputs = inputs.float()
        outputs = distance_model(inputs)[1]
        distance_outputs = np.concatenate((distance_outputs,outputs.detach().numpy()),axis = 0)
        #print(distance_outputs.shape)
    
    # concatence
    x = np.concatenate((distance_outputs[1:],time["pickup_datetime"].values.reshape(-1,1)),axis = 1)
    y = time["trip_period"].values
    y = (y-min(y))/(max(y)-min(y))
    y = y.reshape(-1,1)
    time_ds = TensorDataset(torch.from_numpy(x),torch.from_numpy(y))

    print(time_ds)
    # 分割成训练集和预测集
    n_train = int(len(x)*0.8)
    n_valid = len(x) - n_train
    time_ds_train,time_ds_valid = random_split(time_ds,[n_train,n_valid])
    time_loader_train,time_loader_valid = DataLoader(time_ds_train,batch_size= exp.batch_size),DataLoader(time_ds_valid,batch_size=exp.batch_size)

    # model
    time_model = TimeModule()

    # optimizer
    optimizer = torch.optim.SGD(time_model.parameters(),lr = exp.lr,momentum=exp.momenta, weight_decay=exp.weight_decay)

    # loss function
    criterion = torch.nn.MSELoss(reduce = True,size_average = True)

    return time_model,optimizer,criterion,time_loader_train,time_loader_valid

    
def time_train(exp,time_model,optimizer,criterion,time_loader_train,time_loader_valid,device = "cpu",eval_metric = "MSE"):
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    columns = ["epoch","samples","train_loss","valid_loss"]
    log = pd.DataFrame(columns=columns)

    time_model.to(device)
    time_model.train()
    for epoch in range(exp.epochs):
        for i,(inputs,labels) in enumerate(time_loader_train):
            inputs,labels = inputs.to(device),labels.to(device)
            inputs = inputs.float()
            outputs = time_model(inputs)
            #print(outputs.size(),labels.size())
            loss = criterion(outputs,labels.float())
            #print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1)%100 ==0:
                eval_criterion = torch.nn.MSELoss(reduce = True,size_average = True) if eval_metric=="MSE" else torch.nn.L1Loss(size_average=True,reduce=True,reduction='mean')
                valid_loss = time_evaluate(time_model,eval_criterion,time_loader_valid)
                print("In epoch {} and {} samples, train loss is {}, valid loss is {} ".format((epoch+1),(i+1),loss.item(),valid_loss))
                log = pd.DataFrame({"epoch":[epoch+1],"samples":[i+1],"train_loss":[loss.item()],"valid_loss_"+eval_metric:[valid_loss]},columns = ["epoch","samples","train_loss","valid_loss"])
                path = "./log/time_"+timestamp+".csv"
                if i+1==100:
                    log.to_csv(path,mode="a",index = False,header = True)
                else:
                    log.to_csv(path,mode="a",index = False,header = False)
                time_model.train()
                
    
    # save the model
    path = "./models/time_"+timestamp+".pth"
    torch.save(time_model.state_dict(),path)
    print("训练完成")

def time_evaluate(time_model,criterion,time_loader_valid,device = "cpu"):
    time_model.eval()
    total,total_loss = len(time_loader_valid),0
    for i,(inputs,labels) in enumerate(time_loader_valid):
        inputs,labels = inputs.to(device),labels.to(device)
        inputs = inputs.float()

        outputs = time_model(inputs)
        loss = criterion(outputs,labels.float())
        if not np.isnan(loss.item()):
            total_loss += loss.item()/total
            #print(total_loss,loss.item(),total)
    
    return total_loss

if __name__=="__main__":
    torch.set_default_tensor_type(torch.FloatTensor)
    random_seed()
    exp = experiment()
    #distance_model,optimizer,criterion,distance_loader_train,distance_loader_valid = init_distance(exp)
    #distance_train(exp,distance_model,optimizer,criterion,distance_loader_train,distance_loader_valid,device = "cpu")
    #valid_loss = distance_evaluate(distance_model,criterion,distance_loader_valid)
    #print(valid_loss)
    #distance_model = DistModule()
    #time_model,optimizer,criterion,time_loader_train,time_loader_valid = init_time(exp,distance_model)
    #time_train(time_model,optimizer,criterion,time_loader_train,time_loader_valid,eval_metric = "MAE")
