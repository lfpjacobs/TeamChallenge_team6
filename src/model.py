"""
This code is adapted from the paper of Phillip Isola, et al. in their 2016 paper titled 
"Image-to-Image Translation with Conditional Adversarial Networks"

Available from: https://github.com/phillipi/pix2pix
"""

import tensorflow as tf
import numpy as np
import sys
if "" not in sys.path : sys.path.append("")
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization

def define_discriminator(image_shape):
    """
    This function implements the 70×70 PatchGAN discriminator model as per the 
    design of the model in the paper. 
    
    Input - Two (concatenated) input images 
    Output - Patch output of predictions
    """
    # weight initialization
    init = RandomNormal(stddev=0.02)
	# source image input
    in_src_image = Input(shape=image_shape)
	# target image input
    in_target_image = Input(shape=image_shape)
	# concatenate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image])
	# C64
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
	# C128
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
	# C256
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
	# C512
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
	# patch output
    d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
	# define model
    model = Model([in_src_image, in_target_image], patch_out)
	# compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model

def define_encoder_block(layer_in, n_filters, batchnorm=True):
    """
    Create blocks of layers for the encoder
    """
	# weight initialization
    init = RandomNormal(stddev=0.02)
	# add downsampling layer
    g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
	# leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g

def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    """
    Create blocks of layers for the decoder
    """
	# weight initialization
    init = RandomNormal(stddev=0.02)
	# add upsampling layer
    g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
    g = BatchNormalization()(g, training=True)
	# conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
	# merge with skip connection
    g = Concatenate()([g, skip_in])
	# relu activation
    g = Activation('relu')(g)
    return g

def define_generator(image_shape=(256,256,3)):
    """
    Implement the U-Net encoder-decoder generator model
    """
	# weight initialization
    init = RandomNormal(stddev=0.02)
	# image input
    in_image = Input(shape=image_shape)
	# encoder model
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    e7 = define_encoder_block(e6, 512)
	# bottleneck, no batch norm and relu
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
    b = Activation('relu')(b)
	# decoder model
    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)
    g = Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)
	# define model
    model = Model(in_image, out_image)
    return model

def DSSIM_loss(y_true, y_pred):
    """
    Additional generator regularization term for objective function
    
    This function is based on the standard SSIM implementation from: 
    Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). 
    Image quality assessment: from error visibility to structural similarity. 
    IEEE transactions on image processing.
    """
#    return 1 - tf.image.ssim(y_true, y_pred, max_val=1.0)[0] # DSSIM = 1 - SSIM
    return 1 - np.mean(tf.image.ssim(y_true, y_pred, max_val=1.0)) # DSSIM = 1 - SSIM

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
    """
    Connects the generator and discriminator into a composet model and 
    implementing the corresponding loss functions.
    """
	# make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
	# define the source image
    in_src = Input(shape=image_shape)
	# connect the source image to the generator input
    gen_out = g_model(in_src)
	# connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and classification output
        
    model = Model(in_src, [dis_out, gen_out, gen_out])

	# compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae', DSSIM_loss], optimizer=opt, loss_weights=[1, 10, 10])
    return model