import base64
import io
import pickle
import re
import boto3
import json
import os
import requests

from tqdm import trange
# curl -X POST "https://api-cn.faceplusplus.com/facepp/v3/detect" 
# -F "api_key=<api_key>" \ 
# -F "api_secret=<api_secret>" \ 
# -F "image_file=@image_file.jpg" \ 
# -F "return_landmark=1" \ 
# -F "return_attributes=gender,age"


def detect_faces(basepath,dirpath,file):
    if not os.path.exists(os.path.join(basepath, 'facepp_api', file)):
        # print("quote")
        """Detects faces in an image."""
        path = os.path.join(dirpath, file)

        with io.open(path, 'rb') as image:
            data = {'api_key':'_5FIJ5HcL3L5IQTfEEmAvRYbjL6QzGWb',
            'api_secret':'aEGVrumV8O0pACkQf-giP2R_3mcMTF9q',
            'image_base64':base64.b64encode(image.read()),
            'return_attributes':'emotion'}
            r = requests.post(url = 'https://api-us.faceplusplus.com/facepp/v3/detect', data = data)
            responses = r.text
            responses = json.loads(responses)
            
            if len(responses['faces']) != 0:
                data = [{
                    'anger' : responses['faces'][0]['attributes']['emotion']['anger'],
                    'disgust' : responses['faces'][0]['attributes']['emotion']['disgust'],
                    'fear' : responses['faces'][0]['attributes']['emotion']['fear'],
                    'happiness' : responses['faces'][0]['attributes']['emotion']['happiness'],
                    'neutral' : responses['faces'][0]['attributes']['emotion']['neutral'],
                    'sadness' : responses['faces'][0]['attributes']['emotion']['sadness'],
                    'surprise' : responses['faces'][0]['attributes']['emotion']['surprise'],   
                }]
            
                data = json.dumps(data)

            else:
                # 'HAPPY'|'SAD'|'ANGRY'|'CONFUSED'|'DISGUSTED'|'SURPRISED'|'CALM'|'UNKNOWN'|'FEAR',
                data = [{
                    'UNDETECTED' : 1
                }]
                print('UNDETECTED')
                data = json.dumps(data)

            with open(os.path.join(basepath, 'facepp_api', file), mode='w') as f:
                f.write(data)
           
            

try:        
    os.mkdir(os.path.join('/data/jc/data/image/ferplus_hapi', 'facepp_api'))
except:
    pass
for label in range(7):
    basepath = '/data/jc/data/image/ferplus_hapi'
    path = os.path.join(basepath, 'train',str(label))        

    files = [f for f in os.listdir(path)
                        if os.path.isfile(os.path.join(path, f))]
    for i in trange(len(files), desc='requesting amazon api', leave=True):        
        detect_faces(basepath,path,files[i])        
        

    path = os.path.join(basepath,'valid',str(label))
   
    files = [f for f in os.listdir(path)
                        if os.path.isfile(os.path.join(path, f))]
    for i in trange(len(files), desc='requesting amazon api', leave=True):        
        detect_faces(basepath,path,files[i])      