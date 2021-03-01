from data_preperation import data_prep
from model_util import define_discriminator, define_generator, define_gan
from train_util import train

# TODO: @Sjors Fatsoenlijke pipeline in elkaar flansen

def main():
    """
    Main function for the pipeline used for model training and result prediction
    """

    # Load data
    dataDir = "data\\preprocessed\\"
    dataset_train = data_prep(dataDir, True, "train")
    dataset_test = data_prep(dataDir, True, "test")

    image_shape = dataset_train[0].shape[1:]

    # Define the models
    d_model = define_discriminator(image_shape)
    g_model = define_generator(image_shape)

    # Define the composite model
    gan_model = define_gan(g_model, d_model, image_shape)

    ## TRAIN MODEL
    train(d_model, g_model, gan_model, dataset)

    return

if __name__ == "__main__":
    main()