import sys
if "" not in sys.path : sys.path.append("")

import os
from glob import glob
from shutil import copyfile
from tqdm import tqdm
import time


def preprocess_data(datadir, verbose=False):
    """
    This function gets the raw data and the FSL results and restructures them in a convenient data format.
    """

    dataDir = os.path.abspath(datadir)

    if verbose: 
        print(f"Performing data preprocessing in directory:\n{dataDir:s}")
        time.sleep(1)

    # Define current data locations
    rawDir = os.path.join(dataDir, "raw")
    fslDir = os.path.join(dataDir, "FSL_results")
    
    # Get day_0, day_4 masks and images
    day0_images = glob(os.path.join(rawDir, "rat*_dwib0_1_bet.nii.gz"))
    day4_images = glob(os.path.join(fslDir, "rat*", "bet_flirt.nii.gz"))
    day0_masks = glob(os.path.join(rawDir, "rat*_adc1f_lesionmask.nii.gz"))
    day4_masks = glob(os.path.join(fslDir, "rat*", "mask_flirt.nii.gz"))

    # Define new data locations
    newDir = os.path.join(dataDir, "preprocessed")

    # Copy and rename masks and images to new directory
    day0_images_new = [os.path.join(newDir, os.path.split(old_path)[-1][:5], "day0_img.nii.gz") for old_path in day0_images]
    day4_images_new = [os.path.join(newDir, os.path.split(os.path.dirname(old_path))[-1], "day4_img.nii.gz") for old_path in day4_images]
    day0_masks_new = [os.path.join(newDir, os.path.split(old_path)[-1][:5], "day0_mask.nii.gz") for old_path in day0_masks]
    day4_masks_new = [os.path.join(newDir, os.path.split(os.path.dirname(old_path))[-1], "day4_mask.nii.gz") for old_path in day4_masks]

    # Create needed directories and copy files into appropriate locations
    if not os.path.isdir(newDir):
        os.mkdir(newDir)

    # Check whether subject directory exists and create it if not
    for i in range(len(day0_images_new)):
        subjectDir = os.path.dirname(day0_images_new[i])
        if not os.path.isdir(subjectDir):
            os.mkdir(subjectDir)

    # Iteratively copy all files into the new subject directory
    img_paths = range(len(day0_images_new))
    for i in (tqdm(img_paths, ascii=True) if verbose else img_paths):
        copyfile(day0_images[i], day0_images_new[i])
        copyfile(day4_images[i], day4_images_new[i])
        copyfile(day0_masks[i], day0_masks_new[i])
        copyfile(day4_masks[i], day4_masks_new[i])

    return


if __name__ == '__main__':
    preprocess_data("data")