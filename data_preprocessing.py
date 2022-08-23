
import json
import os
import random
import sys

from typing import Iterator, Mapping

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

from typing import Any, NamedTuple

from PIL import Image
from numpy import asarray
import jax.numpy as jnp
import jax
from jax import jit
from jax import random

import transferlearningresnet as resnetModel

# from google.colab import drive
# drive.mount('/content/gdrive')

def wordDictionary(path):
  tokens_series = pd.read_csv(path, delimiter='\s+', header=0, names=['Token', 'Frequency'], usecols=[0]).squeeze()
  # tokens_series.index = tokens_series.index+1
  tokens = {tokens_series.values[i]: tokens_series.index[i] for i in range(tokens_series.shape[0])}
  return tokens


def to_sequence_Array(set_info, InceptionV3_InputImageDim, tokenDic, dataPath):

  """Builds a SequenceExample proto for an outfit.
  """
  set_id = set_info['set_id']
  image_data = []
  image_ids = []
  caption_data = []
  caption_ids = []
  
  dict_len = len(tokenDic)

  outfit_path = os.path.join(dataPath, 'Outfits', str(set_id))

  if not(os.path.exists(outfit_path)):
    return
    
  def fixedLength_ImageAndCaption_Seq(img_caption_Seq, fixedSeqLength):
    imgSeq, captionSeq = img_caption_Seq
    assert len(imgSeq) == len(captionSeq)
    
    l = len(imgSeq)
    if l > fixedSeqLength:
      return imgSeq[:fixedSeqLength], captionSeq[:fixedSeqLength]
    elif l < fixedSeqLength:
      return imgSeq+[imgSeq[-1]]*(fixedSeqLength-l), captionSeq+[captionSeq[-1]]*(fixedSeqLength-l)
    else:
      return imgSeq, captionSeq

  def imageProcessing(filename):
    image = Image.open(filename)
    img_resized = image.resize(InceptionV3_InputImageDim)
    img_numpy =  asarray(img_resized)
    return img_numpy

  def captionProcessing(caption):

    def oneHotEncoding(tokenIds, vectorLength):
      zeroVector = [0]*vectorLength
      oneHotVectors = [zeroVector[:i]+[1]+zeroVector[i+1:] for i in tokenIds]
      oneHotVectorsArray = np.array(oneHotVectors, dtype=jnp.float32)
      return oneHotVectorsArray

    token_ids = [tokenDic.get(word, dict_len) for word in caption.split()]
    if not(len(token_ids)):
      token_ids = [dict_len]
    oneHotVectors = oneHotEncoding(token_ids, dict_len+1)
    oneHotEmbedding = np.sum(oneHotVectors, axis=0)
    return oneHotEmbedding
  
  for image_info in set_info['items']:
    
    imagefile = os.path.join(outfit_path, str(image_info['index']) + '.jpg')
    image = imageProcessing(imagefile)
    image_data.append(image)

    caption = image_info['name']
    oneHotEncoding = captionProcessing(caption)
    caption_data.append(oneHotEncoding)

  image_data, caption_data = fixedLength_ImageAndCaption_Seq((image_data, caption_data), 8)

  caption_feature = np.array(caption_data)

  # Feature extraction using transfer learning (resnet)
  image_np_array = np.array(image_data)
  model = resnetModel.getResNetModel(image_np_array.shape[1:])
  image_feature = model.predict(image_np_array)
  
  return (image_feature, caption_feature)


def to_sequence_Tensor(outfits, InceptionV3_InputImageDim, tokenDic, path):

  outfitSequencesImage = []
  outfitSequencesCaption = []

  for outfit in outfits:
    img_caption_seq = to_sequence_Array(outfit, InceptionV3_InputImageDim, tokenDic, path)
    if img_caption_seq:
      image_seq, caption_seq = img_caption_seq
      outfitSequencesImage.append(image_seq)
      outfitSequencesCaption.append(caption_seq)

  return {'outfitSequencesImage' : jnp.array(outfitSequencesImage, dtype=jnp.float32), 'outfitSequencesCaption' : jnp.array(outfitSequencesCaption, dtype=jnp.float32)}
