import shutil
import os
 
file_source = '/data/jc/data/image/RAFDB/train/'
file_destination = '/data/jc/data/image/RAFDB/train/google_api/'
os.mkdir(file_source+'/google_api/')
for label in range(7):

    for root, dirs, files in os.walk(os.path.join(file_source,str(label),'google_api/')):
        for img_name in files:
            shutil.move(os.path.join(file_source,str(label),'google_api/',img_name), file_destination)
            
    os.rmdir(os.path.join(file_source,str(label),'google_api/'))  