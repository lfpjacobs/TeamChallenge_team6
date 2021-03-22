import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import nibabel as nib
from glob import glob
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from skimage.metrics import structural_similarity

def getDSC(testImage, resultImage):
    """
    Helper function:
    Compute the Dice Similarity Coefficient for image pair.
    Input needs to be 3D array. 
    """

    testArray   = testImage.flatten()
    resultArray = resultImage.flatten()
    
    # similarity = 1.0 - dissimilarity
    return 1.0 - scipy.spatial.distance.dice(testArray, resultArray) 


def get_fnirt_DSC(datadir):
    """
    Calculates dice similarity coefficients between FNIRT masks and
    ground truth lesionmasks. 
    Input: data directory
    Output: list of DSCs for each subject
    """  
    DSC_list=[]
    #Load data directories 
    fsl_subjectDirs = glob(os.path.join(datadir, "FSL_results", "rat*"))
    gt_subjectDirs = glob(os.path.join(datadir, "preprocessed", "rat*"))
    if len(fsl_subjectDirs) != len(gt_subjectDirs):
        raise ValueError("Mismatch in number of subjects")
    
    #Loop over subjects and calculate DSC for fnirt & ground truth pairs
    for subject_n in range(len(fsl_subjectDirs)):
        fsl_subjectDir = fsl_subjectDirs[subject_n]
        gt_subjectDir = gt_subjectDirs[subject_n]
        
        fsl_mask = nib.load(os.path.join(fsl_subjectDir, "mask_fnirt.nii.gz"))
        gt_mask  = nib.load(os.path.join(gt_subjectDir,  "day0_mask.nii.gz"))
        fsl_mask_array = fsl_mask.get_fdata()
        gt_mask_array = gt_mask.get_fdata()
        
        #Make FNIRT masks binary
        threshold_indices = fsl_mask_array != 0
        fsl_mask_array[threshold_indices] = 1
        
        #Calculate and DSC and append 
        DSC = getDSC(fsl_mask_array, gt_mask_array)
        #DSC = getDSC(fsl_mask_array[:,11:13,:], gt_mask_array[:,11:13,:]) #TODO: implement slices
        DSC_list.append(DSC)
    
    return DSC_list

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