# Letter 5 & 6: Expression
# 					AF = afraid 2
# 					AN = angry 0
# 					DI = disgusted 1
# 					HA = happy 3
# 					NE = neutral 6
# 					SA = sad 4
# 					SU = surprised 5

# rafdb:  hapi amazon 
# 0:  Anger  ANGER
# 1:  Disgust DISGUSTED
# 2:  Fear  FEAR
# 3: Happiness / HAPPY
# 4: Sadness / SAD
# 5:  Surprise SURPRISED
# 6: Neutral / CALM


import base64
import io
import pickle
import re
import shutil
import boto3
import json
import os
import requests

from random import sample


from tqdm import trange

file_source = '/data/jc/data/image/KDEF_and_AKDEF/new_KDEF'
file_dest = '/data/jc/data/image/KDEF_and_AKDEF/KDEF_spilit'
file_list = [f for f in os.listdir(file_source)
                        if os.path.isfile(os.path.join(file_source, f))]


for label in range(7):
    try:
        os.mkdir(os.path.join(file_dest,'train',str(label)))
        os.mkdir(os.path.join(file_dest,'valid',str(label)))
    except:
        pass

for label in range(7):
    label_path = os.path.join(file_source, str(label))
    file_name = os.listdir(label_path)
    valid = sample(file_name, int(len(file_name)*0.2))
    train = list(set(file_name) - set(valid))
    for file in valid:
        shutil.move(os.path.join(label_path,file), os.path.join(file_dest,'valid',str(label),file))
    for file in train:
        shutil.move(os.path.join(label_path,file), os.path.join(file_dest,'train',str(label),file))
    
    
# for label in range(7):
#     try:
#         os.mkdir(os.path.join(file_source,str(label)))
#     except:
#         pass
# for file_name in file_list:
#     expression = file_name[4:6]
#     match expression:
#         case 'AF':
#             shutil.move(os.path.join(file_source,file_name), os.path.join(file_source,'2',file_name))
#         case 'AN':
#             shutil.move(os.path.join(file_source,file_name), os.path.join(file_source,'0',file_name))
#         case 'DI':
#             shutil.move(os.path.join(file_source,file_name), os.path.join(file_source,'1',file_name))
#         case 'HA':
#             shutil.move(os.path.join(file_source,file_name), os.path.join(file_source,'3',file_name))
#         case 'NE':
#             shutil.move(os.path.join(file_source,file_name), os.path.join(file_source,'6',file_name))
#         case 'SA':
#             shutil.move(os.path.join(file_source,file_name), os.path.join(file_source,'4',file_name))
#         case 'SU':
#             shutil.move(os.path.join(file_source,file_name), os.path.join(file_source,'5',file_name))
#         case _:
#             raise ValueError('Unknown expression')
            
    
