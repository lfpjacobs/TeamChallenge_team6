import os
import numpy as np
import nibabel as nib
from keras.preprocessing.image import img_to_array
from glob import glob
import random


def data_prep(datadir, split_dataset=False, train_or_test="", split_factor=0.8, random_seed=1234):
    """
    Main data preperation function.
    Note that currently, all training data is stored in memory. TODO: Might want to do this differently after upscaling
    Additionally, the model is trained on 2D slices of the data. TODO: Might want to do this differently
    Also, only DWI-B0 images are taken into account.
    """

    # Predefine lists for the source and target images
    src_images, tar_images = [], []
    
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

    # Iteratively loop over subjects and extract data
    for subject_n in range(len(subjectDirs)):

        subjectDir = subjectDirs[subject_n]
        print(f"Extracting data for subject '{os.path.split(subjectDir)[-1]}' ({subject_n+1}/{len(subjectDirs)})...\t", end="")

        img_src = nib.load(os.path.join(subjectDir, "day0_img.nii.gz"))
        img_tar = nib.load(os.path.join(subjectDir, "day4_img.nii.gz"))

        img_src = img_to_array(img_src.dataobj)
        img_tar = img_to_array(img_tar.dataobj)

        # Padding images to 256x256 size and transposing axes. Assuming ori image shape to be (244,25,244).
        # Final image shape will thus be 256,256,25
        img_src = np.transpose(img_src, (0,2,1))
        img_tar = np.transpose(img_tar, (0,2,1))

        ori_x, ori_y, ori_z = np.shape(img_src)
        new_x, new_y, new_z = (256, 256, ori_z)

        img_src_pad = img_tar_pad = np.zeros((new_x, new_y, new_z))

        img_src_pad[(new_x-ori_x)//2:ori_x+(new_x-ori_x)//2, (new_y-ori_y)//2:ori_y+(new_y-ori_y)//2, :] = img_src[:]
        img_tar_pad[(new_x-ori_x)//2:ori_x+(new_x-ori_x)//2, (new_y-ori_y)//2:ori_y+(new_y-ori_y)//2, :] = img_tar[:]

        # Add individual slices to image lists
        for slice_i in range(new_z):
            src_images.append(img_src_pad[:,:,slice_i])
            tar_images.append(img_tar_pad[:,:,slice_i])
        
        print("Completed")
    
    # Shuffle slices (coherently) and then store them in numpy arrays
    indices = list(range(len(src_images)))
    random.shuffle(indices)

    src_images_shuffled = [src_images[i] for i in indices]
    tar_images_shuffled = [tar_images[i] for i in indices]

    src_array = np.array(src_images_shuffled)
    tar_array = np.array(tar_images_shuffled)

    print(f"\nCompleted data extraction!\nFound a total of {len(src_images)} slices")
    
    return [src_array, tar_array]


if __name__ == "__main__":
    [src_array, tar_array] = data_prep(os.path.join("data","preprocessed"), True, "test")
