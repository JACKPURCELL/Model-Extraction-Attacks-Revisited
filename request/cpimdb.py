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
    info_lb = torch.ones(10000000,dtype=torch.long)*(-1)
    info_conf = torch.zeros(10000000)
    for i in range(len(predictions[dic])):
        hapi_id = int(predictions[dic][i]['example_id'].split('_')[0])*100 + int(predictions[dic][i]['example_id'].split('_')[1])
        info_lb[hapi_id] = torch.tensor((predictions[dic][i]['predicted_label']))
        info_conf[hapi_id] = torch.tensor((predictions[dic][i]['confidence']))
    return info_lb,info_conf
    
if __name__ == '__main__':
    # info_lb1,info_conf1 = gettensor('sa/imdb/amazon_sa/20-03-22')
    # info_lb2,info_conf2 = gettensor('sa/imdb/amazon_sa/21-02-21')
    info_lb1,info_conf1 = gettensor('sa/imdb/amazon_sa/21-02-21')
    info_lb2,info_conf2 = gettensor('sa/imdb/amazon_sa/22-05-23')
    # 22-05-23
    record = 0
    recordcon = 0
    total = 0
    sum = 0

    for i in range(10000000):
        if info_lb1[i] != -1:
            if info_lb1[i] == info_lb2[i]:
                record += 1
                total += 1
                sum += info_conf2[i] - info_conf1[i]
                if info_conf1[i] == info_conf2[i]:
                    recordcon += 1
                    
            else:
                total += 1
            
            

                    
    print("record:",record,"total:",total,"percent",record/total)
    print("recordcon:",recordcon,"total:",total,"percent",recordcon/total)
    print("sum",sum,"avg: ",sum/total)
                    
                    