import json
import os

from tqdm import trange
def detect_faces(dirpath,file):
    if not os.path.exists(os.path.join(dirpath, 'google_api', file)):
        """Detects faces in an image."""
        from google.cloud import vision
        import io
        client = vision.ImageAnnotatorClient()
        path = os.path.join(dirpath, file)

        
        with io.open(path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        response = client.face_detection(image=image)
        faces = response.face_annotations

        # Names of likelihood from google.cloud.vision.enums
        likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                        'LIKELY', 'VERY_LIKELY')
        # print('Faces:')

        for face in faces:
            data = [{
            'joy_likelihood' : likelihood_name[face.joy_likelihood],
            'sorrow_likelihood' : likelihood_name[face.sorrow_likelihood],
            'anger_likelihood' : likelihood_name[face.anger_likelihood],
            'surprise_likelihood' : likelihood_name[face.surprise_likelihood],
            'under_exposed_likelihood' : likelihood_name[face.under_exposed_likelihood],
            'blurred_likelihood' : likelihood_name[face.blurred_likelihood],
            'headwear_likelihood' : likelihood_name[face.headwear_likelihood],   
            }]
            google_return = json.dumps(data)

            with open(os.path.join(dirpath, 'google_api', file), mode='w') as f:
                f.write(google_return)



    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
        
for label in range(7):
    path = os.path.join('/data/jc/data/image/RAFDB/train', str(label))        
    os.mkdir(os.path.join(path, 'google_api'))
    files = [f for f in os.listdir(path)
                        if os.path.isfile(os.path.join(path, f))]
    for i in trange(len(files), desc='requesting google api', leave=True):        
        detect_faces(path,files[i])        
    
    path = os.path.join('/data/jc/data/image/RAFDB/valid', str(label))
    try:        
        os.mkdir(os.path.join(path, 'google_api'))
    except:
        pass
    files = [f for f in os.listdir(path)
                        if os.path.isfile(os.path.join(path, f))]
    for i in trange(len(files), desc='requesting google api', leave=True):        
        detect_faces(path,files[i])      