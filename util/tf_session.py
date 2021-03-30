import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf

def setup_tf_session():
  """
  Function that fixes memory problems for gpu runs
  """ 

  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
      print(e)
  
  return len(gpus)