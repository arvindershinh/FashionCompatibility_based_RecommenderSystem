
import jax.numpy as jnp
import numpy as np

# from google.colab import drive
# drive.mount('/content/gdrive')

# import outfitprocessing as data

from keras.models import Model
from keras.layers import AvgPool2D
from keras.layers import Flatten
from keras.applications.resnet import ResNet50
import itertools
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# resnetInput = np.array(data.a['img_seq']) # (10, 299, 299, 3)
# resnetInput = np.array(data.sequence_Tensor['outfitSequencesImage'])[0]
# (8, 224, 224, 3)

def getResNetModel(inputShape):
  resnet_model = ResNet50(input_shape=inputShape, weights='imagenet', include_top=False)

  # Make all layers non-trainable
  for layer in resnet_model.layers:
      layer.trainable = False

  output_conv5_block3_2_bn = resnet_model.layers[-6].output
  output_avgPool = AvgPool2D(pool_size=(7, 7))(output_conv5_block3_2_bn)
  output_flatten = Flatten()(output_avgPool)

  model = Model(inputs = resnet_model.input, outputs = output_flatten)

  return model
