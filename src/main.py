import sys
if "" not in sys.path : sys.path.append("")
if "src" not in sys.path : sys.path.append("src")

import os
from data_preperation import data_prep
from preprocessing import preprocess_data
from model import define_discriminator, define_generator, define_gan
from training import train
from evaluation import evaluate, get_fsl_metrics, resp_vec_correlation, plot_corr
from util.tf_session import setup_tf_session


def main(dataDir):
  """
  This function runs through the entire training and evaluation process.
  Please note that this functionality is also performed, and elaborated upon, in the notebooks.
  """
  # Setup the tf session for possible gpu usage
  setup_tf_session()

  # Preprocess data
  print("Step 0: Preprocessing data...\t", end="", flush=True)
  preprocess_data(dataDir)
  print("Completed!\n")

  # Load data
  print("Step 1: Loading and extracting data...\n")

  print("Dataset - TRAIN")
  dataset_train, train_subjects = data_prep(os.path.join(dataDir, "preprocessed"), True, "train")
  print("Dataset - TEST")
  dataset_test, test_subjects = data_prep(os.path.join(dataDir, "preprocessed"), True, "test")

  image_shape = dataset_train[0].shape[1:]
  image_shape = (image_shape[0], image_shape[1], 1)

  print("Completed data loading!\n")

  # Define the models
  print("Step 2: Defining models")
  d_model = define_discriminator(image_shape)
  g_model = define_generator(image_shape)

  gan_model = define_gan(g_model, d_model, image_shape)

  print("Defining models completed!\n")

  # Train model
  print("Step 3: Training")
  time = train(d_model, g_model, gan_model, dataset_train)
  print("Training completed!\n")

  # Choose which model (based on step) you want to evaluate (e.g. "0029400")

  # type the following into your prompt (with specified path to logs): 
  # tensorboard --logdir "../logs" 
  # and go to http://localhost:6006/

  print("Step 4: Evaluation")
  eval_SSIMs = evaluate(d_model, g_model, gan_model, dataset_test, time, specific_model="last")
  fsl_SSIM, fsl_DSC = get_fsl_metrics(dataDir, test_subjects)
  resp_vec_cor = resp_vec_correlation(dataDir, test_subjects, eval_SSIMs)
  print("Evaluation completed!\n")


if __name__ == '__main__':
  main("data")
