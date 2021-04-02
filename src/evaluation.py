import sys
if "" not in sys.path : sys.path.append("")

import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
import os
import warnings
import nibabel as nib
import pandas as pd
import seaborn as sns
from keras.models import load_model
from skimage.metrics import structural_similarity


def calc_def_scores(SSIM_list, subject_list, verbose=False):
    """
    This function simply calculates deformation scores based on SSIM scores.
    It's a very simple function, since it's just a linear transformation.
    """

    # Perform linear transform from [1 - 0 (-1)] to [0, 1]
    def_list_slice = [(1 - SSIM) for SSIM in SSIM_list]

    # Calculate deformation per subject
    def_list_subject = [np.mean(def_list_slice[i:i+3]) for i in range(len(subject_list))]

    # Print results (if applicable)
    if verbose:
        for i in range(len(subject_list)):
            print(f"Deformation for subject {subject_list[i][-2:]}:\t{def_list_subject[i]:.4f} [0-1]")

    return def_list_subject


def getDSC(testImage, resultImage):
    """
    Helper function:
    Compute the Dice Similarity Coefficient for image pair.
    Input: image pair (array), can be 3D image
    Output: DSC
    """
    # Collapse array
    testArray   = testImage.flatten()
    resultArray = resultImage.flatten()
    
    # Similarity = 1.0 - dissimilarity
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return 1.0 - scipy.spatial.distance.dice(testArray, resultArray) 


def resp_vec_correlation(datadir, subject_list, SSIM_list, method = 'pearson', show_fig=True):
    """
    Calculates correlation matrix for response vectors, FSL DSC & SSIM, and 
    cGAN resulting SSIM for given subjects. Method is either 'pearson',
    'kendall', 'spearman' or callable.
    Input: directory containing the data, subjects to be correlated (test set), GAN SSIM scores
    Output: dataframe with correlations (matrix)
    """
    # Store subjects for correctly parsing dataframe
    subjects=[]
    for subject in range(len(subject_list)):
        subjects.append(subject_list[subject][-2:])

    # Read response vectors from csv file into dataframe and parse
    filename = os.path.join(datadir, "responsevecs_TC20analysis210113.csv")
    resp_vec = pd.read_csv(filename, sep=',', header=0, index_col=0)

    resp_vec['ID'].apply(int)
    subjects = [int(subject) for subject in subjects]
    resp_vec = resp_vec.loc[resp_vec['ID'].isin(subjects)]

    #Calculate FSL dice scores and SSIM results. Add as columns to dataframe
    SSIM_fsl, DSCs = get_fsl_metrics(datadir, sorted(subject_list)) 

    #Add individual slice SSIMs
    resp_vec_copy = resp_vec[:0]
    for row in range(len(resp_vec)):
        for i in range(3):
            resp_vec_copy = resp_vec_copy.append(resp_vec[row:row+1])
    resp_vec_copy['SSIM - GAN'] = SSIM_list
    resp_vec_copy['DSC - FSL'] = DSCs
    resp_vec_copy['SSIM - FSL'] = SSIM_fsl
    
    #Calculate  and return the correlation matrix
    resp_vec_def = resp_vec_copy.drop('ID', axis=1)
    correlations = resp_vec_def.corr(method)

    if show_fig : plot_corr(resp_vec_def, correlations)

    return correlations

def plot_corr(dataframe, correlations):
    """
    This function generates a figure to visualize the correlations
    """
    
    # Define relevant data
    relevant_data = ["SSIM - GAN", "DSC - FSL", "SSIM - FSL", "ischHLV", "prepostT1pd_CORE", "postNDS_sum"]

    # Plot figure
    dim = len(relevant_data * 5)
    plt.figure(figsize=(dim, dim))

    color_scale = sns.color_palette("YlOrRd", 100)

    for i in range(len(relevant_data)**2):
        plt.subplot(len(relevant_data), len(relevant_data), i+1)

        x_pos = (i % (len(relevant_data))) 
        y_pos = i // len(relevant_data)
        x_data = relevant_data[x_pos]
        y_data = relevant_data[y_pos]

        if x_pos != y_pos and not (x_pos > 2 and y_pos > 2):
            correlation = correlations[x_data][y_data]  
            sns.scatterplot(x=x_data, y=y_data, data=dataframe, color=".1")
            sns.regplot(x=x_data, y=y_data, data=dataframe, scatter=False, color=color_scale[min(int(abs(correlation)*150) - 1, 99)])
        else:
            plt.xticks([], [])
            plt.yticks([], [])

    plt.tight_layout()
    plt.show()
    
    return


def get_fsl_metrics(datadir, subject_list):
    """
    Calculates both dice similarity coefficients between FNIRT masks and
    ground truth lesionmasks as well as SSIM between FNIRT and FLIRT images. 
    Input: data directory, subjects to calculate DSC and SSIM
    Output: list of DSCs and SSIM for each slice of each subject
    """  
    # Initialize output lists
    DSC_list=[]
    SSIM_list=[]
    
    # Load data directories
    fsl_subjectDirs=[]
    gt_subjectDirs=[]
    for subject in range(len(subject_list)):
        subject_id = subject_list[subject][-5:]
        fsl_subjectDirs.append(os.path.join(datadir, "FSL_results", subject_id))
        gt_subjectDirs.append(os.path.join(datadir, "preprocessed", subject_id))
    if len(fsl_subjectDirs) != len(gt_subjectDirs):
        raise ValueError("Mismatch in number of subjects")
    
    # Loop over subjects and calculate DSC for fnirt & ground truth pairs
    for subject_n in range(len(fsl_subjectDirs)):
        fsl_subjectDir = fsl_subjectDirs[subject_n]
        
        # Load images and extract data in array
        fnirt_mask = nib.load(os.path.join(fsl_subjectDir, "mask_fnirt.nii.gz"))
        flirt_mask = nib.load(os.path.join(fsl_subjectDir, "mask_flirt.nii.gz"))
        fnirt_img  = nib.load(os.path.join(fsl_subjectDir, "bet_fnirt.nii.gz"))
        flirt_img  = nib.load(os.path.join(fsl_subjectDir, "bet_flirt.nii.gz"))
        fnirt_mask_array = fnirt_mask.get_fdata()
        flirt_mask_array = flirt_mask.get_fdata()
        fnirt_img_array = fnirt_img.get_fdata()
        flirt_img_array = flirt_img.get_fdata()

        
        # Make FLIRT/FNIRT masks binary
        threshold_indices = fnirt_mask_array != 0
        fnirt_mask_array[threshold_indices] = 1
        threshold_indices = flirt_mask_array != 0
        flirt_mask_array[threshold_indices] = 1
        
        # Calculate DSC and SSIM then append 
        for i in range(3):
            DSC = getDSC(fnirt_mask_array[:,11+i,:], flirt_mask_array[:,11+i,:])
            SSIM = structural_similarity(flirt_img_array[:,11+i,:], fnirt_img_array[:,11+i,:])
            DSC_list.append(DSC)
            SSIM_list.append(SSIM)
        
    return SSIM_list, DSC_list


def evaluate(d_model, g_model, gan_model, dataset, subject_list, time, specific_model="last", verbose=False, show_fig=False):
    """"
    Evaluation function for trained GAN
    input: generator model & test set image pairs
    output: registered test images & DSC
    """
    # If applicable, load a specific model
    if specific_model != "last":
        model_dir = os.path.join("models", f"run_{time}")

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            g_model = load_model(os.path.join(model_dir, 'g_model_{}.h5'.format(specific_model))) # e.g. 0029400  
    
    # Preprocess test set
    true_day4, true_day0 = dataset 
    
    # Generate registered (fake) day0 images
    predicted_day0 = g_model.predict(true_day4.reshape(len(true_day4), 256, 256, 1)) # needs to be 4D
    
    # Plot the results
    SSIM_list = []
    if show_fig : fig, ax = plt.subplots(len(true_day4), 4, figsize=(14, len(true_day4)*3))
    for i in range(len(true_day4)):
        pred_d0 = predicted_day0[i].reshape(256,256)
        true_d0 = np.float32(true_day0[i].reshape(256,256))
        true_d4 = np.float32(true_day4[i].reshape(256,256))
        
        SSIM, _ = structural_similarity(pred_d0, true_d0, full=True)
        _, ssim_map = structural_similarity(pred_d0, true_d4, full=True)

        if show_fig:
            ax[i,0].imshow(pred_d0, cmap='gray')
            ax[i,0].axis('off')
            ax[i,0].set_title('Fake day 0 - rat {}, slice {}'.format(subject_list[i//3][-2:], (i%3) + 1))
            
            ax[i,1].imshow(true_d0, cmap='gray')
            ax[i,1].axis('off')
            ax[i,1].set_title('True day 0 - rat {}, slice {}'.format(subject_list[i//3][-2:], (i%3) + 1))
            
            ax[i,2].imshow(true_d4, cmap='gray')
            ax[i,2].axis('off')
            ax[i,2].set_title('True day 4 - rat {}, slice {}'.format(subject_list[i//3][-2:], (i%3) + 1))

            ax[i,3].imshow(0.5*(1 - ssim_map), cmap='gray')
            ax[i,3].axis('off')
            ax[i,3].set_title('Pred deformation map - rat {}, slice {}'.format(subject_list[i//3][-2:], (i%3) + 1))

        # Quantified structural deformation
        SSIM_def = structural_similarity(pred_d0, true_d4) 
        SSIM_list.append(SSIM_def)
        
        if verbose:
            print("SSIM (day0 v day0_pred) for slice {}: ".format(i), SSIM)     # you want this to be 1
            print("SSIM (day4 v day0_pred) for slice {}: ".format(i), SSIM_def) # deformation
        
    return SSIM_list
      