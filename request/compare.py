import torch
import hapi        
hapi_data_dir = "/home/jkl6486/HAPI"  
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
     
hapi.config.data_dir = hapi_data_dir
label_num = 7
data_num=0
def gettensor(dic):
    dic_split = dic.split('/')
    predictions =  hapi.get_predictions(task=dic_split[0], dataset=dic_split[1], date=dic_split[3], api=dic_split[2])
    info_lb={}
    # info_conf={}

    
    modes = ['train', 'test']
    for mode in modes:
        info_lb[mode] = torch.ones(15000)*(-1)
        # info_conf[mode] = torch.zeros(20000)
        
        for i in range(len(predictions[dic])):
            hapi_mode = predictions[dic][i]['example_id'].split('_')[0]
            hapi_id = int(predictions[dic][i]['example_id'].split('_')[1])
            if hapi_mode == mode:
                info_lb[mode][hapi_id] = torch.tensor((predictions[dic][i]['confidence']))
    return info_lb

    
if __name__ == '__main__':
    dic1 = gettensor('fer/rafdb/microsoft_fer/20-03-05')
    dic2 = gettensor('fer/rafdb/microsoft_fer/21-02-16')
    record = 0
    total = 0
    modes = ['train']
    for mode in modes:
        for label_1,label_2 in zip(dic1[mode],dic2[mode]):
            if label_1 != -1:
                if label_1 == label_2:
                    record += 1
                    total += 1
                else:
                    total += 1
                    
    print("record:",record,"total:",total,"percent",record/total)
                    
                    