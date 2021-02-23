from data_preperation import data_prep
from model_util import define_discriminator, define_generator, define_gan
from train_util import train


#%% LOAD STUFF
path_day0 = "C:\\Users\\20166218\\Documents\\Master\\Q3\\Team Challenge\\Image analysis\\Data\\DWI\\day0"
path_day4 = "C:\\Users\\20166218\\Documents\\Master\\Q3\\Team Challenge\\Image analysis\\Data\\DWI\\day4"

dataset = data_prep(path_day0, path_day4)
print('Loaded', dataset[0].shape, dataset[1].shape)
image_shape = dataset[0].shape[1:]

# define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)


#%% TRAIN MODEL
train(d_model, g_model, gan_model, dataset)