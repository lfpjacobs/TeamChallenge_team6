import os
import numpy as np
import nibabel as nib
import random
import ndimage
from keras.preprocessing.image import img_to_array

def data_prep(path_day0, path_day4):
    """
    Input paths to dwi images of day0 and day4
    list of source and target images are returned
    """

    src_images, tar_images = [], []
    
    angles_list = range(0,360,72) # (360/72)*22 = 110, 100 epochs --> 11000 training steps
    
    for filename in os.listdir(path_day0):
        f = os.path.join(path_day0, filename)
        img = nib.load(f)
        img = img_to_array(img.dataobj)
        background = np.zeros((256,256,25)) # multiple of two dimensions...
        background[:224, :224, :] = np.transpose(img, (0,2,1))
        for a in angles_list:
            img_rot = ndimage.rotate(background[:,:,10:11], a, reshape=False)
            #src_images.append(background[:,:,10:11]) #random slice
            img_rot = np.float32(img_rot)
            src_images.append(img_rot) #random 3 slices
       
    for filename in os.listdir(path_day4):
        f = os.path.join(path_day4, filename)
        img = nib.load(f)
        img = img_to_array(img.dataobj)
        background = np.zeros((256,256,25)) # multiple of two dimensions...
        background[:224, :224, :] = np.transpose(img, (0,2,1))
        for a in angles_list:
            img_rot = ndimage.rotate(background[:,:,10:11], a, reshape=False)
            img_rot = np.float32(img_rot)
            #tar_images.append(background[:,:,10:11]) #random 3 slices
            tar_images.append(img_rot) #random 3 slices
    
    c = list(zip(src_images, tar_images))
    random.shuffle(c)
    src_images, tar_images = zip(*c)
    
    src_list = np.array(src_images)
    tar_list = np.array(tar_images)
    
    return [src_list, tar_list]