import os
import numpy as np
import tensorflow as tf
import random
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from augmentation import augment
from datetime import datetime
from skimage.metrics import structural_similarity

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

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	# unpack dataset
    trainA, trainB = dataset
	# choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
	# retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
	# generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y


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
def summarize_performance(step, g_model, dataset_train, dataset_test, modelsDir, logger, run, n_samples=3):
	# select a sample of input images
    [X_realA_train, X_realB_train], _ = generate_real_samples(dataset_train, n_samples, 1)
	# generate a batch of fake samples
    X_fakeB_train, _ = generate_fake_samples(g_model, X_realA_train, 1)

    # select a sample of input images
    [X_realA_test, X_realB_test], _ = generate_real_samples(dataset_test, n_samples, 1)
	# generate a batch of fake samples
    X_fakeB_test, _ = generate_fake_samples(g_model, X_realA_test, 1)
	
	# save the generator model
    filename = os.path.join(modelsDir,'g_model_{:07d}.h5'.format((step+1)))
    g_model.save(filename)
    print('>Saved model: {}s'.format(filename))
    
    logger.log_images('run_{}_step{}_train'.format(run, step), [X_realA_train[0], X_fakeB_train[0], X_realB_train[0]], step)
    logger.log_images('run_{}_step{}_val'.format(run, step), [X_realA_test[0], X_fakeB_test[0], X_realB_test[0]], step)


def check_ssim(g_model, dataset, n_samples=3):
    # select a sample of input images
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	# generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    
    SSIM = structural_similarity(np.float32(X_realB.reshape(256,256)), X_fakeB.reshape(256,256))
    
    return SSIM

# train pix2pix model
def train(d_model, g_model, gan_model, dataset_train, dataset_test, n_epochs=100, n_batch=1):    
    # Extract current time for model/plot save files
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    modelsDir = os.path.join("..", "..", "models", f"run_{current_time}")
    os.mkdir(modelsDir)

    logsDir = os.path.join("..", "..", "logs", f"run_{current_time}")
    os.mkdir(logsDir)

	# determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
	
    n_aug = 20 # number of augmentations per epoch (so after step 22*5=110, i.e. epoch 1, the images are freshly augmented)
    # value of 5 is just for ease of testing
    # 22 training images, n_aug = 2 --> 44 (newly augmented) training images each epoch
        
    trainA_ori, trainB_ori = dataset_train
    testA, testB = dataset_test

    # Fix train- and test-dataset dimensionality issues
    trainA = np.expand_dims(trainA_ori, axis=3)
    trainB = np.expand_dims(trainB_ori, axis=3)
    testA = np.expand_dims(testA, axis=3)
    testB = np.expand_dims(testB, axis=3)

    dataset_train = [trainA, trainB]
    dataset_test = [testA, testB]

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
    for i in range(n_steps):
        if (i) % (bat_per_epo) == 0: # per epoch "refresh" training set with new augmentations
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
                    # filename_realA = os.path.join("..", "..", "data", "aug_debug", str(i)+"_"+str(n)+"_"+str(j)+"_realA.png")
                    # filename_A = os.path.join("..", "..", "data", "aug_debug", str(i)+"_"+str(n)+"_"+str(j)+"_A.png")
                    # filename_B = os.path.join("..", "..", "data", "aug_debug", str(i)+"_"+str(n)+"_"+str(j)+"_B.png")
                    # debug_img_realA = Image.fromarray((255*trainA_ori[j].reshape((256, 256))).astype(np.uint8))
                    # debug_imgA = Image.fromarray((255*trainA_aug.reshape((256, 256))).astype(np.uint8))
                    # debug_imgB = Image.fromarray((255*trainB_aug.reshape((256, 256))).astype(np.uint8))
                    # debug_img_realA.save(filename_realA)
                    # debug_imgA.save(filename_A)
                    # debug_imgB.save(filename_B)
                    
            dataset_aug = [A, B] # all trainingdata (day4 and day1)
            print("Completed")

        # select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(dataset_aug, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB, X_realB])
        # summarize performance
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))

        # summarize model performance and store models
        if (i+1) % (bat_per_epo) == 0:
            summarize_performance(i, g_model, dataset_train, dataset_test, modelsDir, logger_im, current_time)
        
        # Store losses (tensorboard) 
        if (i+1) % (bat_per_epo // 50) == 0:
            logger_g.log_scalar('run_{}'.format(current_time), g_loss, i)
            logger_d1.log_scalar('run_{}'.format(current_time), d_loss1, i)
            logger_d2.log_scalar('run_{}'.format(current_time), d_loss2, i)  

        # Store similarities (tensorboard)
        if (i+1) % (bat_per_epo // 20) == 0:
            neg_similarity_train = check_ssim(g_model, dataset_train, 1)
            neg_similarity_val = check_ssim(g_model, dataset_test, 1) # should probably select one validation slice instead
            
            logger_train.log_scalar('run_{}'.format(current_time), neg_similarity_train, i) # higher is better (more similar)
            logger_val.log_scalar('run_{}'.format(current_time), neg_similarity_val, i)
    
    return current_time