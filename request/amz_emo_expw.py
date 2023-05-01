import io
import pickle
import re
import boto3
import json
import os

from tqdm import trange
client = boto3.client('rekognition')
def detect_faces(dirpath,file):
    if not os.path.exists(os.path.join(dirpath, 'amazon_api', file)):
        # print("quote")
        """Detects faces in an image."""
        path = os.path.join(dirpath, file)

        with io.open(path, 'rb') as image:
            responses = client.detect_faces(Image={'Bytes': image.read()},Attributes= [ "ALL" ])
            if len(responses['FaceDetails']) != 0:
                data = [{
                    responses['FaceDetails'][0]['Emotions'][0]["Type"] : responses['FaceDetails'][0]['Emotions'][0]["Confidence"],
                    responses['FaceDetails'][0]['Emotions'][1]["Type"] : responses['FaceDetails'][0]['Emotions'][1]["Confidence"],
                    responses['FaceDetails'][0]['Emotions'][2]["Type"] : responses['FaceDetails'][0]['Emotions'][2]["Confidence"],
                    responses['FaceDetails'][0]['Emotions'][3]["Type"] : responses['FaceDetails'][0]['Emotions'][3]["Confidence"],
                    responses['FaceDetails'][0]['Emotions'][4]["Type"] : responses['FaceDetails'][0]['Emotions'][4]["Confidence"],
                    responses['FaceDetails'][0]['Emotions'][5]["Type"] : responses['FaceDetails'][0]['Emotions'][5]["Confidence"],
                    responses['FaceDetails'][0]['Emotions'][6]["Type"] : responses['FaceDetails'][0]['Emotions'][6]["Confidence"],
                    responses['FaceDetails'][0]['Emotions'][7]["Type"] : responses['FaceDetails'][0]['Emotions'][7]["Confidence"]
                    
                }]
            
                data = json.dumps(data)
                origin = json.dumps(responses['FaceDetails'][0])
            else:
                # 'HAPPY'|'SAD'|'ANGRY'|'CONFUSED'|'DISGUSTED'|'SURPRISED'|'CALM'|'UNKNOWN'|'FEAR',
                data = [{
                    'UNDETECTED' : 1
                }]
            
                data = json.dumps(data)
                origin = 'null'
            with open(os.path.join(dirpath, 'amazon_api', file), mode='w') as f:
                f.write(data)
            with open(os.path.join(dirpath, 'amazon_api_origin', file), mode='w') as f:
                f.write(origin)
            


for label in range(7):
    path = os.path.join('/data/jc/data/image/EXPW_224/train', str(label))        
    try:        
        os.mkdir(os.path.join(path, 'amazon_api'))
        os.mkdir(os.path.join(path, 'amazon_api_origin'))
    except:
        pass
    files = [f for f in os.listdir(path)
                        if os.path.isfile(os.path.join(path, f))]
    for i in trange(len(files), desc='requesting amazon api', leave=True):        
        detect_faces(path,files[i])        
    
    path = os.path.join('/data/jc/data/image/EXPW_224/valid', str(label))
    try:        
        os.mkdir(os.path.join(path, 'amazon_api'))
        os.mkdir(os.path.join(path, 'amazon_api_origin'))
    except:
        pass
    files = [f for f in os.listdir(path)
                        if os.path.isfile(os.path.join(path, f))]
    for i in trange(len(files), desc='requesting amazon api', leave=True):        
        detect_faces(path,files[i])      