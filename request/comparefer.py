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
# def gettensor(dic):
#     dic_split = dic.split('/')
#     predictions =  hapi.get_predictions(task=dic_split[0], dataset=dic_split[1], date=dic_split[3], api=dic_split[2])
#     lb = hapi.get_labels(task=dic_split[0], dataset=dic_split[1])
#     info_lb = torch.ones(100000)*(-1)
    
#     for i in range(len(predictions[dic])):
#         hapi_id = int(predictions[dic][i]['example_id'].split('fer')[1])
#         info_lb[hapi_id] = torch.tensor((predictions[dic][i]['predicted_label']))
#     return info_lb
    
    
    
# if __name__ == '__main__':
#     dic1 = gettensor('fer/ferplus/microsoft_fer/20-03-05')
#     dic2 = gettensor('fer/ferplus/microsoft_fer/21-02-15')
#     record = 0
#     total = 0


#     for label_1,label_2 in zip(dic1,dic2):
#         if label_1 != -1:
#             if label_1 == label_2:
#                 record += 1
#                 total += 1
#             else:
#                 total += 1
                    
#     print("record:",record,"total:",total,"percent",record/total)
                    
                    
def gettensor(dic):
    dic_split = dic.split('/')
    predictions =  hapi.get_predictions(task=dic_split[0], dataset=dic_split[1], date=dic_split[3], api=dic_split[2])
    
    info_lb = torch.ones(100000)*(-1)
    
    for i in range(len(predictions[dic])):
        hapi_id = int(predictions[dic][i]['example_id'].split('fer')[1])
        info_lb[hapi_id] = torch.tensor((predictions[dic][i]['predicted_label']))
    return info_lb
    
    
    
if __name__ == '__main__':
    # dic1 = gettensor('fer/ferplus/microsoft_fer/20-03-05')
    dic1 = gettensor('fer/ferplus/microsoft_fer/21-02-15')
    lb = hapi.get_labels(task='fer', dataset='ferplus')['fer/ferplus']
    lb_info_map = torch.ones(100000)*(-1)
    
    for i in range(len(lb)):
        hapi_id = int(lb[i]['example_id'].split('fer')[1])
        lb_info_map[hapi_id] = torch.tensor((lb[i]['true_label']))
        
    record = 0
    total = 0


    for label_1,label_2 in zip(dic1,lb_info_map):
        if label_1 != -1:
            if label_1 == label_2:
                record += 1
                total += 1
            else:
                total += 1
                    
    print("record:",record,"total:",total,"percent",record/total)
                    
                                        