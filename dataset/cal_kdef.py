import os

import torch
from kdef import KDEF
from torch.utils.data import DataLoader

from rafdb import RAFDB


def get_mean_std(trainLoader):
    imgs = None
    for batch in trainLoader:
        image_batch = batch[0]
        if imgs is None:
            imgs = image_batch.cpu()
        else:
            imgs = torch.cat([imgs, image_batch.cpu()], dim=0)
    imgs = imgs.numpy()
    
    # calculate mean over each channel (r,g,b)
    mean_r = imgs[:,0,:,:].mean()
    mean_g = imgs[:,1,:,:].mean()
    mean_b = imgs[:,2,:,:].mean()
    print(mean_r,mean_g,mean_b)

    # calculate std over each channel (r,g,b)
    std_r = imgs[:,0,:,:].std()
    std_g = imgs[:,1,:,:].std()
    std_b = imgs[:,2,:,:].std()
    print(std_r,std_g,std_b)
    
    
    
# train_dataset = KDEF(input_directory=os.path.join('/data/jc/data/image/KDEF_and_AKDEF/KDEF_spilit',"train"),hapi_data_dir='/home/jkl6486/HAPI',hapi_info='sa/imdb/amazon_sa/22-05-23',api='amazon',transform='Normal')
# test_dataset = KDEF(input_directory=os.path.join('/data/jc/data/image/KDEF_and_AKDEF/KDEF_spilit',"valid"),hapi_data_dir='/home/jkl6486/HAPI',hapi_info='sa/imdb/amazon_sa/22-05-23',api='amazon',transform='Normal')

# train_dataset = RAFDB(input_directory=os.path.join('/data/jc/data/image/RAFDB',"train"),hapi_data_dir='/home/jkl6486/HAPI',hapi_info='fer/rafdb/microsoft_fer/22-05-23',api=None,transform=None)
test_dataset = RAFDB(input_directory=os.path.join('/data/jc/data/image/RAFDB',"valid"),hapi_data_dir='/home/jkl6486/HAPI',hapi_info='fer/rafdb/microsoft_fer/22-05-23',api=None,transform=None)

train_loader = DataLoader(dataset=test_dataset,
                    batch_size=256,
                    shuffle=True,sampler=None,
                    num_workers=8,drop_last=True)

get_mean_std(train_loader)