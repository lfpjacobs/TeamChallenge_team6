import numpy as np
import gryds # https://github.com/tueimage/gryds --> pip install git+https://github.com/tueimage/gryds
import random

# TODO: @Sjors, implement more augmentation, e.g. noise / L/R flip and more deformation/rotation.
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
        angles=[random.uniform(-np.pi/36, np.pi/36)],
        center=[0.5, 0.5]
    )

    # Define an image interpolater
    an_image_interpolatorA = gryds.Interpolator(imA)
    an_image_interpolatorB = gryds.Interpolator(imB)
    
    # Combine all operations and apply the same augmentation to day0 and day4
    composed = gryds.ComposedTransformation(bspline, a_rotation, a_translation)
    transformed_imageA = an_image_interpolatorA.transform(composed)
    transformed_imageB = an_image_interpolatorB.transform(composed)
    
    return [transformed_imageA.reshape((256,256,1)), transformed_imageB.reshape((256,256,1))]