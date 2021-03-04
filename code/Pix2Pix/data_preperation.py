import os
import numpy as np
import nibabel as nib
from keras.preprocessing.image import img_to_array
from glob import glob
import random
import scipy
import matplotlib.pyplot as plt


def generate_importance_map(map_size):
    """
    This small function generates a weighted map of "where we want the brain to be".
    It is thus 1 at the center and 0 at the edges.
    """
    importance_map = np.zeros(map_size)
    for i in range(map_size[0]):
        for j in range(map_size[1]):
            for k in range(map_size[2]):
                importance = np.sum(map_size)//2 - abs(i - map_size[0]//2) - abs(j - map_size[1]//2) - abs(k - map_size[2]//2)
                importance_map[i,j,k] = importance
    
    importance_map = np.abs(importance_map / (np.sum(map_size)//2))

    return importance_map


def optimalize_offsets(img, old_size, crop_size, importance_map):
    """
    This function optimizes the offsets for the cropping algorithm.
    Due to time constraints, it does so by first making a rough estimate of the appropriate x- and y-offsets and then trying
    all combinations in a small range. 
    """
    # Define bounding box position offsets. Define the z-offset to be perfectly centered.
    x_offsets = range(old_size[0] - crop_size[0])
    y_offsets = range(old_size[1] - crop_size[1])
    z_offset = (old_size[2] - crop_size[2]) // 2

    # Loop over possible offsets and determine the best one
    best_offsets = [0, 0]
    img_sum = np.sum(img)

    y_start1, y_start2 = ((old_size[1]-crop_size[1])//2, crop_size[1]+(old_size[1]-crop_size[1])//2)
    best_brainness = 0

    # For computing time purposes, we won't go over all possible combinations of x and y. 
    # Instead, we'll separate x and y to obtain a rough estimate and retry if cropping fails.
    while best_brainness <= 0:

        # x_offset
        best_brainness = 0
        for x_offset in x_offsets:
            try:
                patch = img[x_offset:crop_size[0]+x_offset, y_start1:y_start2, z_offset:crop_size[2]+z_offset]
                brainness = np.sum(patch * importance_map) - 5 * (img_sum - np.sum(patch))
            except ValueError:
                y_start1 = 0
                y_start2 = 0
                patch = img[x_offset:crop_size[0]+x_offset, y_start1:y_start2, z_offset:crop_size[2]+z_offset]
                brainness = np.sum(patch * importance_map) - 5 * (img_sum - np.sum(patch))

            if brainness > best_brainness:
                best_brainness = brainness
                best_offsets[0] = x_offset
        
        # y_offset
        best_brainness = 0
        for y_offset in y_offsets:
            patch = img[best_offsets[0]:crop_size[0]+best_offsets[0], y_offset:crop_size[0]+y_offset, z_offset:crop_size[2]+z_offset]
            brainness = np.sum(patch * importance_map) - 5 * (img_sum - np.sum(patch))

            if brainness > best_brainness:
                best_brainness = brainness
                best_offsets[1] = y_offset
        
        # If necessary, update y starting points
        y_start1 += 10
        y_start2 += 10
    
    # Now, we will check all possible combinations of x and y in a smaller range to obtain a better estimate
    narrow_x_offsets = range(best_offsets[0]-5, best_offsets[0]+5)
    narrow_y_offsets = range(best_offsets[1]-5, best_offsets[1]+5)

    for x_offset in narrow_x_offsets:
        for y_offset in narrow_y_offsets:
            patch = img[x_offset:crop_size[0]+x_offset, y_offset:crop_size[0]+y_offset, z_offset:crop_size[2]+z_offset]
            brainness = np.sum(patch * importance_map) - 5 * (img_sum - np.sum(patch))

            if brainness > best_brainness:
                best_brainness = brainness
                best_offsets = [x_offset, y_offset]
    
    return best_offsets, z_offset


def crop_brain(src_img, tar_img, subject_name, importance_map, crop_size=(128, 128, 25)):
    """
    This function crops the input images to a specified size and maximizes
    the presence of brain tissue in the bounding box.
    It is used to remove some of the unnecessary zero padding around the BET-treated images
    and thus improve model performance.
    """
    if np.shape(src_img) == np.shape(tar_img):
        old_size = np.shape(src_img)
    else:
        raise ValueError("The source and target image aren't the same size!")

    # Optimize x- and y-offsets to centralize the brain
    best_offsets, z_offset = optimalize_offsets(src_img, old_size, crop_size, importance_map)

    # Generate cropped images based on optimal offsets
    src_img_crop = src_img[best_offsets[0]:crop_size[0]+best_offsets[0], best_offsets[1]:crop_size[1]+best_offsets[1], z_offset:crop_size[2]+z_offset]
    tar_img_crop = tar_img[best_offsets[0]:crop_size[0]+best_offsets[0], best_offsets[1]:crop_size[1]+best_offsets[1], z_offset:crop_size[2]+z_offset]

    # Resample the images back to the original image size (256x256 in our case)
    src_img_res = scipy.ndimage.zoom(src_img_crop, (old_size[0]//crop_size[0], old_size[1]//crop_size[1], 1), order=1)
    tar_img_res = scipy.ndimage.zoom(tar_img_crop, (old_size[0]//crop_size[0], old_size[1]//crop_size[1], 1), order=1)

    # Flip images into a more readable format
    src_img_res = np.flip(src_img_res.swapaxes(0, 1), axis=0)
    tar_img_res = np.flip(tar_img_res.swapaxes(0, 1), axis=0)

    # Make and save figures for debugging purposes
    # Source image
    for i in range(crop_size[2]):
        plt.subplot(int(np.sqrt(crop_size[2]))+1, int(np.sqrt(crop_size[2]))+1 , 1+i)
        plt.axis('off')
        plt.imshow(src_img_res[:,:,i], cmap='gray')
    
    plt.savefig(os.path.join("..", "..", "data", "preprocessed", "brain_extraction",  f"cropped_src_{subject_name}.png"))
    plt.close()

    # Target image
    for i in range(crop_size[2]):
        plt.subplot(int(np.sqrt(crop_size[2]))+1, int(np.sqrt(crop_size[2]))+1 , 1+i)
        plt.axis('off')
        plt.imshow(tar_img_res[:,:,i], cmap='gray')
    
    plt.savefig(os.path.join("..", "..", "data", "preprocessed", "brain_extraction",  f"cropped_tar_{subject_name}.png"))
    plt.close()

    return [src_img_res, tar_img_res]


def data_prep(datadir, split_dataset=False, train_or_test="", split_factor=0.8, random_seed=1234):
    """
    Main data preperation function.
    Note that currently, all training data is stored in memory. TODO: Might want to do this differently after upscaling
    Additionally, the model is trained on 2D slices of the data. TODO: Might want to do this differently
    Also, only DWI-B0 images are taken into account.
    """
    
    # Define required path and (if applicable) make a division for train/test sets
    subjectDirs = glob(os.path.join(datadir, "rat*"))

    if split_dataset:
        random.Random(random_seed).shuffle(subjectDirs)
        if train_or_test.lower() == "train":
            subjectDirs = subjectDirs[:int(len(subjectDirs)*split_factor + 1)]
        elif train_or_test.lower() == "test":
            subjectDirs = subjectDirs[int(len(subjectDirs)*split_factor + 1):]
        else:
            raise ValueError("train_or_test parameter should be either 'train' or 'test'")
    elif not split_dataset:
        pass
    else:
        raise ValueError("split_dataset should be either True or False")

    # Predifine some parameters needed for brain extraction
    crop_size = (128, 128, 21)
    importance_map = generate_importance_map(crop_size)
    # Make brain extraction visualisation directory
    if not os.path.isdir((os.path.join("..", "..", "data", "preprocessed", "brain_extraction"))):
        os.mkdir((os.path.join("..", "..", "data", "preprocessed", "brain_extraction")))

    # initialize data
    src_array = np.zeros((len(subjectDirs)*crop_size[2], 256, 256))
    tar_array = np.zeros((len(subjectDirs)*crop_size[2], 256, 256))
    j = 0
    # Iteratively loop over subjects and extract data
    for subject_n in range(len(subjectDirs)):

        subjectDir = subjectDirs[subject_n]
        print(f"Extracting data for subject '{os.path.split(subjectDir)[-1]}' ({subject_n+1}/{len(subjectDirs)})...\t", end="")

        img_src = nib.load(os.path.join(subjectDir, "day4_img.nii.gz"))
        img_tar = nib.load(os.path.join(subjectDir, "day0_img.nii.gz"))

        img_src = img_to_array(img_src.dataobj)
        img_tar = img_to_array(img_tar.dataobj)

        # Padding images to 256x256 size and transposing axes. Assuming ori image shape to be (244,25,244).
        # Final image shape will thus be 256,256,25
        img_src = np.transpose(img_src, (0,2,1))
        img_tar = np.transpose(img_tar, (0,2,1))

        ori_x, ori_y, ori_z = np.shape(img_src)
        new_x, new_y, new_z = (256, 256, ori_z)

        img_src_pad = np.zeros((new_x, new_y, new_z))
        img_tar_pad = np.zeros((new_x, new_y, new_z))

        img_src_pad[(new_x-ori_x)//2:ori_x+(new_x-ori_x)//2, (new_y-ori_y)//2:ori_y+(new_y-ori_y)//2, :] = img_src[:]
        img_tar_pad[(new_x-ori_x)//2:ori_x+(new_x-ori_x)//2, (new_y-ori_y)//2:ori_y+(new_y-ori_y)//2, :] = img_tar[:]

        # Now, crop the images to remove unnecessary empty space
        [img_src_crop, img_tar_crop] = crop_brain(img_src_pad, img_tar_pad, os.path.split(subjectDir)[-1], importance_map, crop_size)

        # Add individual slices to image lists 
        for slice_i in range(crop_size[2]):
            src_array[j] = img_src_crop[:,:,slice_i]
            tar_array[j] = img_tar_crop[:,:,slice_i]
            j += 1
            
        print("Completed")

    if np.shape(src_array)[2] > 0:
        print(f"\nCompleted data extraction!\nFound a total of {np.shape(src_array)[2]} slices\n")
        if not np.shape(src_array)[2] == np.shape(tar_array)[2] : raise ValueError("There aren't as many source as target images. Check for missing files.")
    else:
        raise ValueError("The selected data directory doesn't contain any properly formatted data")
    
    return [src_array, tar_array]


if __name__ == "__main__":
    [src_array, tar_array] = data_prep(os.path.join("..", "..", "data","preprocessed"), False)
