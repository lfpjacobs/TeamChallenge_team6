import sys
if "" not in sys.path : sys.path.append("")

import os
import numpy as np
import tensorflow as tf
import random
import math
from tqdm import tqdm
import warnings
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from augmentation import augment
from datetime import datetime
from skimage.metrics import structural_similarity
from util.general import print_style

# load and prepare training images
def load_real_samples(filename):
	# load compressed arrays
    data = load(filename)
	# unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]


def generate_real_samples(dataset, n_samples, patch_shape, available_idx=None):
    """
    Function that selects a batch of random samples from a list of available ones, returns images and target.
    """
    # unpack dataset
    trainA, trainB = dataset

    # choose random instances
    if type(available_idx) in [list, np.ndarray]:
        ix = np.random.randint(0, len(available_idx), n_samples)
        img_idx = available_idx[ix]
    else:
        img_idx = np.random.randint(0, np.shape(trainA)[0], n_samples)
        ix = None

    # retrieve selected images
    X1, X2 = trainA[img_idx], trainB[img_idx]
    # generate 'real' class labels (1)
    y = np.ones((n_samples, patch_shape, patch_shape, 1))

    return [X1, X2], y, ix

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
    X = g_model.predict(samples)
	# create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y

# create a logging object to log scalar losses and images to Tensorboard
class Logger(object):
    def __init__(self, log_dir):
        self.writer = tf.summary.create_file_writer(log_dir)

    def log_scalar(self, tag, value, step):
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()
    
    def log_images(self, tag, images, step):
        with self.writer.as_default():
            tf.summary.image(tag, np.asarray(images), max_outputs=len(images), step=step)
            self.writer.flush()
              
# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset_train, dataset_val, modelsDir, logger, run, n_samples=3):
	# select a sample of input images
    [X_realA_train, X_realB_train], _, _ = generate_real_samples(dataset_train, n_samples, 1)
	# generate a batch of fake samples
    X_fakeB_train, _ = generate_fake_samples(g_model, X_realA_train, 1)

    # select a sample of input images
    [X_realA_val, X_realB_val], _, _ = generate_real_samples(dataset_val, n_samples, 1)
	# generate a batch of fake samples
    X_fakeB_val, _ = generate_fake_samples(g_model, X_realA_val, 1)
	
	# save the generator model
    filename = os.path.join(modelsDir,'g_model_{:07d}.h5'.format((step+1)))
    g_model.save(filename)
    print('>Saved model: {}s'.format(filename))
    
    logger.log_images('run_{}_step{}_train'.format(run, step), [X_realA_train[0], X_fakeB_train[0], X_realB_train[0]], step)
    logger.log_images('run_{}_step{}_val'.format(run, step), [X_realA_val[0], X_fakeB_val[0], X_realB_val[0]], step)


def check_ssim(g_model, dataset, n_samples=3):
    # select a sample of input images
    [X_realA, X_realB], _, _ = generate_real_samples(dataset, n_samples, 1)
	# generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    
    SSIM = structural_similarity(np.float32(X_realB.reshape(256,256)), X_fakeB.reshape(256,256))
    
    return SSIM


def split_train_val(dataset, split_factor=0.8):
    """
    This function is used to split the training set into a separate training- and test-set.
    """
    
    dataset_size = np.shape(dataset)
    train_len = math.floor(dataset_size[1]*split_factor)
    val_len = math.floor(dataset_size[1]*(1-split_factor))

    train_size = (dataset_size[0], train_len, dataset_size[2], dataset_size[3])
    val_size = (dataset_size[0], val_len, dataset_size[2], dataset_size[3])

    dataset_train = np.zeros(train_size)
    dataset_val = np.zeros(val_size)

    subject_indices = list(range(dataset_size[1]))
    random.shuffle(subject_indices)

    for i in range(train_len):
        subject_n = subject_indices[i]
        dataset_train[0][i] = dataset[0][subject_n]
        dataset_train[1][i] = dataset[1][subject_n]

    for j in range(val_len):
        subject_m = subject_indices[-(j+1)]
        dataset_val[0][j] = dataset[0][subject_m]
        dataset_val[1][j] = dataset[1][subject_m]

    return dataset_train, dataset_val

# train pix2pix model
def train(d_model, g_model, gan_model, dataset_train, n_epochs=100, n_batch=4):    
    # Extract current time for model/plot save files
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    modelsDir = os.path.join("models", f"run_{current_time}")
    os.mkdir(modelsDir)

    logsDir = os.path.join("logs", f"run_{current_time}")
    os.mkdir(logsDir)

	# determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
	
    n_aug = 20 # number of augmentations per epoch (so after step 22*5=110, i.e. epoch 1, the images are freshly augmented)
    # value of 5 is just for ease of testing
    # 22 training images, n_aug = 2 --> 44 (newly augmented) training images each epoch
    
    # Split training set in train and validation sets
    dataset_train, dataset_val = split_train_val(dataset_train)

    trainA_ori, trainB_ori = dataset_train
    valA, valB = dataset_val

    # Fix train- and test-dataset dimensionality issues
    trainA = np.expand_dims(trainA_ori, axis=3)
    trainB = np.expand_dims(trainB_ori, axis=3)
    valA = np.expand_dims(valA, axis=3)
    valB = np.expand_dims(valB, axis=3)

    dataset_train = [trainA, trainB]
    dataset_val = [valA, valB]

	# calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA_ori)*n_aug / n_batch) # for us: 22*n_aug
	# calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs # for us: 2200*n_aug
	  
    # Define loggers for losses, images and similarity metrics
    logger_g = Logger(os.path.join(logsDir, "gen"))
    logger_d1 = Logger(os.path.join(logsDir, "dis1"))
    logger_d2 = Logger(os.path.join(logsDir, "dis2"))
    logger_im = Logger(os.path.join(logsDir, "im"))
    logger_train = Logger(os.path.join(logsDir, "ssim_train"))
    logger_val = Logger(os.path.join(logsDir, "ssim_val"))
    
    # manually enumerate epochs
    i = 0

    for epoch in range(n_epochs):
        print("\n"+print_style.BOLD+f"Epoch {epoch+1}/{n_epochs}:"+print_style.END)

        # Create list of available indices --> Images can't be used twice in the same epoch.
        available_idx = np.array(range(np.shape(dataset_train)[1]))

        # For each epoch, create a new augmented dataset
        print("Performing data augmentation... ", end="", flush=True) 
        # initialize augmented dataset
        A = np.zeros((len(trainA_ori)*n_aug, 256, 256, 1))
        B = np.zeros((len(trainA_ori)*n_aug, 256, 256, 1))
        # for shuffling of all slices
        rand_i = list(range(len(trainA_ori)*n_aug)) 
        random.shuffle(rand_i) # unique list of random indices
        k = 0
        
        for n in range(n_aug):
            for j in range(len(trainA_ori)):
                # Augment every image randomely per epoch
                trainA_aug, trainB_aug = augment(trainA_ori[j], trainB_ori[j])
                A[rand_i[k]] = trainA_aug
                B[rand_i[k]] = trainB_aug
                k += 1

                # Write images for debugging purposes
                # filename_realA = os.path.join("data", "aug_debug", str(i)+"_"+str(n)+"_"+str(j)+"_realA.png")
                # filename_A = os.path.join("data", "aug_debug", str(i)+"_"+str(n)+"_"+str(j)+"_A.png")
                # filename_B = os.path.join("data", "aug_debug", str(i)+"_"+str(n)+"_"+str(j)+"_B.png")
                # debug_img_realA = Image.fromarray((255*trainA_ori[j].reshape((256, 256))).astype(np.uint8))
                # debug_imgA = Image.fromarray((255*trainA_aug.reshape((256, 256))).astype(np.uint8))
                # debug_imgB = Image.fromarray((255*trainB_aug.reshape((256, 256))).astype(np.uint8))
                # debug_img_realA.save(filename_realA)
                # debug_imgA.save(filename_A)
                # debug_imgB.save(filename_B)
                
        dataset_aug = [A, B] # all trainingdata (day4 and day1)
        print("Completed")

        for batch in tqdm(range(bat_per_epo), ascii=True):           

            # select a batch of real samples
            [X_realA, X_realB], y_real, used_idx = generate_real_samples(dataset_aug, n_batch, n_patch, available_idx)

            np.delete(available_idx, used_idx)

            # generate a batch of fake samples
            X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)

            # Perform actual training
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # update discriminator for real samples
                d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
                # update discriminator for generated samples
                d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
                # update the generator
                g_loss, _, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB, X_realB])               
            
            # Store losses (tensorboard) 
            if (i+1) % (bat_per_epo // 50) == 0:
                logger_g.log_scalar('run_{}'.format(current_time), g_loss, i)
                logger_d1.log_scalar('run_{}'.format(current_time), d_loss1, i)
                logger_d2.log_scalar('run_{}'.format(current_time), d_loss2, i)  

            # Store similarities (tensorboard)
            if (i+1) % (bat_per_epo // 20) == 0:
                similarity_train = check_ssim(g_model, dataset_train, 1)
                similarity_val = check_ssim(g_model, dataset_val, 1) # should probably select one validation slice instead
                
                logger_train.log_scalar('run_{}'.format(current_time), similarity_train, i) # higher is better (more similar)
                logger_val.log_scalar('run_{}'.format(current_time), similarity_val, i)

            i += 1
        
        print('>Losses: d1[%.3f] d2[%.3f] g[%.3f]' % (d_loss1, d_loss2, g_loss))
        summarize_performance(i, g_model, dataset_train, dataset_val, modelsDir, logger_im, current_time)
    
    return current_time