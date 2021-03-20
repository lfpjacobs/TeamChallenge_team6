import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import nibabel as nib
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from skimage.metrics import structural_similarity

# TODO: This code isn't functional yet. Need to take a look at this.

#def getDSC(testImage, resultImage):
#    """
#    Helper function:
#    Compute the Dice Similarity Coefficient for image pair.
#    Input needs to be 3D array 
#    """
#
#    testArray   = testImage.flatten()
#    resultArray = resultImage.flatten()
#    
#    # similarity = 1.0 - dissimilarity
#    return 1.0 - scipy.spatial.distance.dice(testArray, resultArray) 
#
#
#def get_maskDSC(subject_list, g_model, datadir):
#    """
#    Perform registration on mask day 4 --> day0. Then compute the DSC 
#    for all test set pairs.
#    input: list of rat subjects in test set, generator model, datadir
#    output: array with pairwise DSC
#    """
#    DSC_scores=[] #store dice scores here
#    
#    #load the lesionmasks for day0 and day4
#    dataDir = os.path.join("..", "..", "data")
#
#    # Define current data locations
#    PreprocessedDir = os.path.join(dataDir, "preprocessed")
#    
#    #load masks
#    day4_masks=[]    
#    day0_masks_gt=[] #Ground truth day0 masks
#    day0_mask_reg=[] #Registered day0 masks
#    for subject in subject_list:
#        img_src = nib.load(os.path.join(PreprocessedDir, subject, "day4_mask.nii.gz"))
#        img_tar = nib.load(os.path.join(PreprocessedDir, subject, "day0_mask.nii.gz"))
#
#        img_src = img_to_array(img_src.dataobj)
#        img_tar = img_to_array(img_tar.dataobj)
#        day4_masks.append(img_src)
#        day0_masks_gt.append(img_tar)
#    
#    #perform registration on mask day 4 --> day 0
#    for i in range(len(day4_masks)): 
#        day0_mask_reg = g_model.predict(day4_masks[i]) #I am not sure how to do this, this is probably wrong
#        DSC_scores.append(getDSC(day0_masks_gt[i], day0_mask_reg))
#    
#    return DSC_scores

def evaluate(d_model, g_model, gan_model, dataset, specific_model=False):
    """"
    Evaluation function for trained GAN
    input: generator model & test set image pairs
    output: registered test images & DSC
    """
    # load model, gebeurt nu niks mee, maar dit zou in functie parameters kunnen om een specifiek model te evalueren
    if specific_model == True:
        os.path.join("..", "..", "models")
        g_model = load_model('g_model_0029400.h5') #hier iets verzinnen om simpel een model te kiezen (of in main)    
    
    #Preprocess test set
    true_day4, true_day0 = dataset 
    
    #Generate registered (fake) day0 images
    predicted_day0 = g_model.predict(true_day4.reshape(len(true_day4), 256, 256, 1)) # needs to be 4D
    
    #Plot predicted (registered) and true image side by side 
    SSIM_list = []
    fig, ax = plt.subplots(3, len(true_day4), figsize=(25, 5))
    for i in range(len(true_day4)):
        pred = predicted_day0[i].reshape(256,256)
        true = np.float32(true_day0[i].reshape(256,256))
        
        SSIM, ssim_map = structural_similarity(pred, true, full=True)
        
        SSIM_list.append(SSIM)
        print("SSIM of subject {}: ".format(i), SSIM)
        
        ax[0,i].imshow(pred, cmap='gray')
        ax[0,i].axis('off')
        ax[0,i].set_title('Test subject {}'.format(i))
        ax[1,i].imshow(true, cmap='gray')
        ax[1,i].axis('off')
        ax[2,i].imshow(ssim_map, cmap='gray')
        ax[2,i].axis('off')
        
    # first row shows predicted scans
    # second row shows the corresponding ground truths
    # third row shows full structural similarity image (for possible qualitative analysis)
    
    # quantify deformation:
    # eventually compute SSIM between predicted day0 and given day4