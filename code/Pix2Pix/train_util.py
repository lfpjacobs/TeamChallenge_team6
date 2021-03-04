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

# load and prepare training images
def load_real_samples(filename):
	# load compressed arrays
    data = load(filename)
	# unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
    # TODO: @Bas Ff kijken naar of dit de goeie intensiteitsrange is --> Standardisation functie inbouwen
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
def summarize_performance(step, g_model, dataset, modelsDir, logger, run, n_samples=3):
	# select a sample of input images
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	# generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	# scale all pixels from [-1,1] to [0,1]
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0
	
	# save the generator model
    filename = os.path.join(modelsDir,'g_model_{:06d}.h5'.format((step+1)))
    g_model.save(filename)
    print('>Saved model: {}s'.format(filename))
    
    logger.log_images('run_{}_step{}'.format(run, step), [X_realA[0], X_fakeB[0], X_realB[0]], step)   

# train pix2pix model
def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):    
    # Extract current time for model/plot save files
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    modelsDir = os.path.join("..", "..", "models", f"run_{current_time}")
    os.mkdir(modelsDir)

    logsDir = os.path.join("..", "..", "logs", f"run_{current_time}")
    os.mkdir(logsDir)

	# determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
	
    n_aug = 5 # number of augmentations per epoch (so after step 22*5=110, i.e. epoch 1, the images are freshly augmented)
    # value of 5 is just for ease of testing
    # 22 training images, n_aug = 2 --> 44 (newly augmented) training images each epoch
        
    trainA, trainB = dataset
	# calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA)*n_aug / n_batch) # for us: 22*n_aug
	# calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs # for us: 2200*n_aug
	  
    # need three seperate loggers for the three scalar losses
    logger_g = Logger(os.path.join(logsDir, "gen"))
    logger_d1 = Logger(os.path.join(logsDir, "dis1"))
    logger_d2 = Logger(os.path.join(logsDir, "dis2"))
    logger_im = Logger(os.path.join(logsDir, "im"))
    
    # manually enumerate epochs
    for i in range(n_steps):
        if (i) % (bat_per_epo) == 0: # per epoch "refresh" training set with new augmentations 
            # initialize augmented dataset
            A = np.zeros((len(trainA)*n_aug, 256, 256, 1))
            B = np.zeros((len(trainA)*n_aug, 256, 256, 1))
            # for shuffling of all slices
            rand_i = list(range(len(trainA)*n_aug)) 
            random.shuffle(rand_i) # unique list of random indices
            k = 0
            
            for n in range(n_aug):
                for j in range(len(trainA)):
                    # Augment every image randomely per epoch
                    trainA_aug, trainB_aug = augment(trainA[j], trainB[j])
                    A[rand_i[k]] = trainA_aug
                    B[rand_i[k]] = trainB_aug
                    k += 1
                    
            dataset_aug = [A, B] # all trainingdata (day4 and day1)
        
        # select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(dataset_aug, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # summarize performance
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))

        # summarize model performance and store models
        if (i+1) % (bat_per_epo) == 0:
            summarize_performance(i, g_model, dataset_aug, modelsDir, logger_im, current_time)
        
        # Store losses (tensorboard) 
        if (i+1) % (bat_per_epo // 50):
            logger_g.log_scalar('run_{}'.format(current_time), g_loss, i)
            logger_d1.log_scalar('run_{}'.format(current_time), d_loss1, i)
            logger_d2.log_scalar('run_{}'.format(current_time), d_loss2, i)        
            
        # type the following into your prompt: 
        # tensorboard --logdir "logs"
        # and go to http://localhost:6006/