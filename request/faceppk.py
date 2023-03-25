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


def detect_faces(dirpath,file):
    if not os.path.exists(os.path.join('/data/jc/data/image/KDEF_and_AKDEF/facepp_api', file)):
        # print("quote")
        """Detects faces in an image."""
        path = os.path.join(dirpath, file)

        with io.open(path, 'rb') as image:
            data = {'api_key':'0KHi8-QNz1qcDUSAzpcbCSQBfBL8GZPJ',
            'api_secret':'0A_i8hjaXU4UqLbmpXKpj9qmEwzUqBn0',
            'image_base64':base64.b64encode(image.read()),
            'return_attributes':'emotion'}

            r = requests.post(url = 'https://api-cn.faceplusplus.com/facepp/v3/detect', data = data)

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

            with open(os.path.join('/data/jc/data/image/KDEF_and_AKDEF/facepp_api', file), mode='w') as f:
                f.write(data)
           
            


path = '/data/jc/data/image/KDEF_and_AKDEF/KDEF_spilit/valid'       


for label in range(7):
    label_path = os.path.join(path, str(label))
    # try:        
    #     os.mkdir(os.path.join(path, 'facepp_api'))
    # except:
    #     pass

    files = [f for f in os.listdir(label_path)
                        if os.path.isfile(os.path.join(label_path, f))]
    for i in trange(len(files), desc='requesting facepp api', leave=True):        
        detect_faces(label_path,files[i])        

    