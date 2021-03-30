import sys
if "" not in sys.path : sys.path.append("")

import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import os
import nibabel as nib
import pandas as pd
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


def resp_vec_correlation(datadir, subject_list, SSIM_list, method = 'pearson'):
    """
    Calculates correlation matrix for response vectors, DSC and SSIM
    for given subjects. Method is either 'pearson', 'kendall', 'spearman' 
    or callable. 
    """
    #Store subjects for correctly parsing dataframe
    subjects=[]
    for subject in range(len(subject_list)):
        subjects.append(subject_list[subject][-2:])

    #Read response vectors from csv file into dataframe and parse
    filename = os.path.join(datadir, "responsevecs_TC20analysis210113.csv")
    resp_vec = pd.read_csv(filename, sep=',', header=0, index_col=0)
    resp_vec = resp_vec.loc[resp_vec['ID'].isin(subjects)]
    
    #Calculate FNIRT dice scores and add as column to dataframe
    DSCs = get_fnirt_DSC(datadir, sorted(subject_list)) 
    resp_vec['DSC'] = DSCs
    
    #Add individual slice SSIMs
    resp_vec_copy = resp_vec[:0]
    for row in range(len(resp_vec)):
        for i in range(3):
            resp_vec_copy = resp_vec_copy.append(resp_vec[row:row+1])
    resp_vec_copy['SSIM'] = SSIM_list
    
    #Calculate  and return the correlation matrix
    resp_vec_def = resp_vec_copy.drop('ID', axis=1)
    correlations = resp_vec_def.corr(method)
    return correlations


def get_fnirt_DSC(datadir, subject_list):
    """
    Calculates dice similarity coefficients between FNIRT masks and
    ground truth lesionmasks. 
    Input: data directory, subject to calculate DSC
    Output: list of DSCs for each subject
    """  
    DSC_list=[]
    #Load data directories
    fsl_subjectDirs=[]
    gt_subjectDirs=[]
    for subject in range(len(subject_list)):
        subject_id = subject_list[subject][-5:]
        fsl_subjectDirs.append(os.path.join(datadir, "FSL_results", subject_id))
        gt_subjectDirs.append(os.path.join(datadir, "preprocessed", subject_id))
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
        #DSC = getDSC(fsl_mask_array, gt_mask_array)
        DSC = getDSC(fsl_mask_array[:,11:14,:], gt_mask_array[:,11:14,:]) 
        DSC_list.append(DSC)
    
    return DSC_list


def evaluate(d_model, g_model, gan_model, dataset, time, specific_model="last"):
    """"
    Evaluation function for trained GAN
    input: generator model & test set image pairs
    output: registered test images & DSC
    """
    # load model, gebeurt nu niks mee, maar dit zou in functie parameters kunnen om een specifiek model te evalueren
    if specific_model != "last":
        model_dir = os.path.join("models", f"run_{time}")
        g_model = load_model(os.path.join(model_dir, 'g_model_{}.h5'.format(specific_model))) # e.g. 0029400  
    
    #Preprocess test set
    true_day4, true_day0 = dataset 
    
    #Generate registered (fake) day0 images
    predicted_day0 = g_model.predict(true_day4.reshape(len(true_day4), 256, 256, 1)) # needs to be 4D
    
    #Plot results
    SSIM_list = []
    fig, ax = plt.subplots(4, len(true_day4), figsize=(25, 10))
    for i in range(len(true_day4)):
        pred_d0 = predicted_day0[i].reshape(256,256)
        true_d0 = np.float32(true_day0[i].reshape(256,256))
        true_d4 = np.float32(true_day4[i].reshape(256,256))
        
        SSIM, ssim_map = structural_similarity(pred_d0, true_d0, full=True)

        ax[0,i].imshow(pred_d0, cmap='gray')
        ax[0,i].axis('off')
        ax[0,i].set_title('Pred d0 subject {}'.format(i))
        
        ax[1,i].imshow(true_d0, cmap='gray')
        ax[1,i].axis('off')
        ax[1,i].set_title('True d0 subject {}'.format(i))
        
        ax[2,i].imshow(ssim_map, cmap='gray')
        ax[2,i].axis('off')
        ax[2,i].set_title('SSIM map subject {}'.format(i))
        
        ax[3,i].imshow(true_d4, cmap='gray')
        ax[3,i].axis('off')
        ax[3,i].set_title('True d4 subject {}'.format(i))

        # quantified structural deformation
        SSIM_def = structural_similarity(pred_d0, true_d4) 
        SSIM_list.append(SSIM_def)
        
        print("SSIM (day0 v day0_pred) of subject {}: ".format(i), SSIM) # you want this to be 1
        print("SSIM (day4 v day0_pred) of subject {}: ".format(i), SSIM_def) # deformation
        
    return SSIM_list
      