

#!pip install git+https://github.com/deepmind/dm-haiku

#pip install optax

# from google.colab import drive
# drive.mount('/content/gdrive')

import optax
import sys
from math import ceil
import matplotlib.pyplot as plt
#sys.path.insert(0,'/content/gdrive/MyDrive/Colab Notebooks/Capstone Project/Dependencies')

import model_fashion

from typing import Iterator, Mapping
import json
import os
import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
import pickle

def batching(tensor, batch_size):
    batch,sequence,embedding = tensor.shape
    n = int(ceil(batch/batch_size))
    m = (tensor[batch_size*i:batch_size*(i+1),:,:] for i in range(n))
    return m

"""LOAD TRAINING DATA"""

# path_T = '/content/gdrive/MyDrive/Colab Notebooks/Capstone Project/Outfit_Processing/Archive and Analysis/pkl files/TrainingDataset'
path_T = "Data/"

image_np_T = np.load(os.path.join(path_T, 'outfitSequencesImage.npy'), allow_pickle=True)
caption_np_T = np.load(os.path.join(path_T, 'outfitSequencesCaption.npy'), allow_pickle=True)

print("Training data : ", image_np_T.shape, caption_np_T.shape)

batch_size = 100
image_itr_Train = batching(image_np_T, batch_size)
caption_itr_Train = batching(caption_np_T, batch_size)

"""LOAD VALIDATION DATA"""

#path_V = '/content/gdrive/MyDrive/Colab Notebooks/Capstone Project/Outfit_Processing/Archive and Analysis/pkl files/ValidationDataset'
path_V = "Data/"

image_np_V = np.load(os.path.join(path_V, 'outfitSequencesImage_validation.npy'), allow_pickle=True)
caption_np_V = np.load(os.path.join(path_V, 'outfitSequencesCaption_validation.npy'), allow_pickle=True)

print("Validation dataset :", image_np_V.shape, caption_np_V.shape)

image_itr_V = batching(image_np_V, batch_size)
caption_itr_V = batching(caption_np_V, batch_size)

"""TRAINING"""

model = hk.transform(model_fashion.total_loss)
epochs = 50
train_loss_epoch = []
val_loss_epoch = []
def train_model(train_ds, valid_ds) -> hk.Params:
  """Initializes and trains a model on train_ds, returning the final params."""
  global image_np_T
  global caption_np_T
  global image_itr_V
  global caption_itr_V

  rng = jax.random.PRNGKey(428)
  opt = optax.adam(1e-3)

  image_itr_T, caption_itr_T = train_ds
  image_itr_V, caption_itr_V = valid_ds

  @jax.jit
  def loss(params, x):
    pred = model.apply(params, None, x)
    return pred

  @jax.jit
  def update(step, params, opt_state, x):
    l, grads = jax.value_and_grad(loss)(params, x)
    grads, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, grads)
    return l, params, opt_state

  # Initialize state.
  try:
    sample_x = {'outfitSequencesImage': next(image_itr_T), 'outfitSequencesCaption': next(caption_itr_T)}
  except:
    return
  
  params = model.init(rng, sample_x)
  opt_state = opt.init(params)


  steps = range(int((image_np_T.shape[0]/batch_size))+1)
  train_losses = []
  val_losses = []

  for epoch in range(1, epochs + 1):
      val_count = 1
      print("Epoch : ", epoch)
      for step in steps:
        if step % 12 == 0:
          try:
            print("val count : ", val_count)
            x_V = {'outfitSequencesImage': next(image_itr_V), 'outfitSequencesCaption': next(caption_itr_V)}
            val_loss = loss(params, x_V)
            print("Step {}: valid loss {}".format(step, val_loss))
            val_losses.append(val_loss)
            val_count = val_count + 1
          except:
              image_itr_V = batching(image_np_V, batch_size)
              caption_itr_V = batching(caption_np_V, batch_size)

        try:
            x_T = {'outfitSequencesImage': next(image_itr_T), 'outfitSequencesCaption': next(caption_itr_T)}
            train_loss, params, opt_state = update(step, params, opt_state, x_T)
            train_losses.append(train_loss)
            print("Step {}: train loss {}".format(step, train_loss))
        except:
            image_itr_T = batching(image_np_T, batch_size)
            caption_itr_T = batching(caption_np_T, batch_size)
      train_loss_epoch.append(train_loss)
      val_loss_epoch.append(val_loss)
      print("Epoch : ", epoch, " | train_loss : ",train_loss_epoch, " | validation_loss : ", val_loss_epoch)
  return params, train_losses, val_losses


from datetime import datetime
now = datetime.now()
params, train_loss, val_loss = train_model((image_itr_Train, caption_itr_Train), (image_itr_V, caption_itr_V))
print(params.keys())
print("started at =", now)
now = datetime.now()
print("ending at =", now)

import matplotlib.pyplot as plt

def my_plot(epochs, t_loss, v_loss):
    plt.plot(epochs, t_loss, label='train_loss')
    plt.plot(epochs, v_loss, label='val_loss')
    plt.legend()
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
my_plot(np.linspace(1, epochs, epochs).astype(int), train_loss_epoch, val_loss_epoch)

params_path = "params.p"
pickle.dump(params, open(params_path, "wb"))
print("params ready at : ", params_path)