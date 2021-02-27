import os
import numpy as np
import nibabel as nib
from keras.preprocessing.image import img_to_array

def data_prep(path_day0, path_day4):
    """
    Input paths to dwi images of day0 and day4
    list of source and target images are returned
    """
    
    # All 22 images are just used for training for now
    
    src_images, tar_images = [], []
    
    for filename in os.listdir(path_day0):
        f = os.path.join(path_day0, filename)
        img = nib.load(f)
        img = img_to_array(img.dataobj)
        
        # I got dimensionality errors when using 224x224 images, so I added a 
        # background to let our images have the same dimensionality as theirs 
        # (must possibly be a multiple of 2?)
        background = np.zeros((256,256,25))
        background[:224, :224, :] = np.transpose(img, (0,2,1))
        # gryds (used for augmentation) has implemented different border modes 
        # which we can explore as well
        
        img = np.float32(background[:,:,10:11]) # random slice (2D problem for now)
        src_images.append(img)
       
    for filename in os.listdir(path_day4):
        f = os.path.join(path_day4, filename)
        img = nib.load(f)
        img = img_to_array(img.dataobj)
       
        background = np.zeros((256,256,25))
        background[:224, :224, :] = np.transpose(img, (0,2,1))
        
        img = np.float32(background[:,:,10:11])
        tar_images.append(img)
    
    src_list = np.array(src_images)
    tar_list = np.array(tar_images)
    
    return [src_list, tar_list]