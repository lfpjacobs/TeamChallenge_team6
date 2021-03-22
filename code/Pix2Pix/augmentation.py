import numpy as np
import gryds # https://github.com/tueimage/gryds --> pip install git+https://github.com/tueimage/gryds
import random


def augment(imA, imB):
    """
    Input day0 and day4 image of the same rat
    An augmented version of both is returned
    """
    
    imA = imA.reshape((256,256)) #"3D" to 2D
    imB = imB.reshape((256,256))
    
    # Define random deformation matrix
    random_grid = np.random.rand(2, 3, 3) 
    random_grid -= 0.5
    random_grid /= 10
    
    bspline = gryds.BSplineTransformation(random_grid)
    
    # Define random translation matrix
    a_translation = gryds.TranslationTransformation([random.uniform(-0.03, 0.03), random.uniform(-0.03, 0.03)])
        
    # Define random rotation matrix
    a_rotation = gryds.AffineTransformation(
        ndim=2,
        angles=[random.uniform(-np.pi/24, np.pi/24)],
        center=[0.5, 0.5]
    )

    # Define an image interpolater
    an_image_interpolatorA = gryds.Interpolator(imA)
    an_image_interpolatorB = gryds.Interpolator(imB)
    
    # Combine all operations and apply the same augmentation to day0 and day4
    composed = gryds.ComposedTransformation(bspline, a_rotation, a_translation)
    transformed_imageA = an_image_interpolatorA.transform(composed)
    transformed_imageB = an_image_interpolatorB.transform(composed)
    
    # Define noise augmentation
    mu = 0.0
    sigma = random.uniform(0., 0.05)
    noise_mapA = np.random.normal(mu, sigma, size = np.size(imA)).reshape((256, 256))
    noise_mapB = np.random.normal(mu, sigma, size = np.size(imB)).reshape((256, 256))
    noise_mapA[transformed_imageA < 1e-2] = 0.
    noise_mapB[transformed_imageB < 1e-2] = 0.

    transformed_imageA = transformed_imageA + noise_mapA
    transformed_imageB = transformed_imageB + noise_mapB

    # Flip L/R (half of the time)
    perform_flip = random.choice([False, True])
    if perform_flip:
        transformed_imageA = np.fliplr(transformed_imageA)
        transformed_imageB = np.fliplr(transformed_imageB)

    return [transformed_imageA.reshape((256,256,1)), transformed_imageB.reshape((256,256,1))]