

# !pip install git+https://github.com/deepmind/dm-haiku

import jax.numpy as jnp


from typing import Iterator, Mapping

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from typing import Any, NamedTuple

import haiku as hk
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds
from functools import partial

lstm_embedding_size = 256
num_lstm_units = 256
rng = jax.random.PRNGKey(42)
dropout_rate = 0.2
# batch_norm = hk.BatchNorm(False, False, 0.99)
def Linear_Model():
  model = hk.Linear(lstm_embedding_size, with_bias=False, b_init=None)
  return model

def LSTM_Model(is_training:bool) -> hk.RNNCore:
  """Defines the network architecture."""
  if is_training:
    dropout_rate = 0.2
  else:
    dropout_rate = 0
  model = hk.DeepRNN([
      hk.LSTM(num_lstm_units),
      partial(hk.dropout, rng, dropout_rate),
      hk.LSTM(num_lstm_units),
      partial(hk.dropout, rng, dropout_rate),
      hk.Linear(lstm_embedding_size, with_bias=False, b_init=None),
      hk.nets.MLP([lstm_embedding_size])
  ])
  return model

def sequence_loss(batch) -> jnp.ndarray:
  """Unrolls the network over a sequence of inputs & targets, gets loss."""
  # Note: this function is impure; we hk.transform() it below.

  sequence_length, batch_size = batch.shape[:-1]

  batch_NonSeq = jnp.reshape(batch, (-1, batch.shape[-1]))
  linear = Linear_Model()
  batch_embedding = linear(batch_NonSeq)

  batch_Input_F = jnp.reshape(batch_embedding, (sequence_length, batch_size, -1))
  batch_Input_B = jnp.flip(batch_Input_F, 0)

  coreF = LSTM_Model(True)
  coreB = LSTM_Model(True)

  initial_state1 = coreF.initial_state(batch_size)
  initial_state2 = coreB.initial_state(batch_size)

  # shape = (S,B,E)
  logitsF, _ = hk.dynamic_unroll(coreF, batch_Input_F, initial_state1)
  logitsB, _ = hk.dynamic_unroll(coreB, batch_Input_B, initial_state2)

  # shape = (S*B,E)
  logitsF_Flatten = jnp.reshape(logitsF, (-1, logitsF.shape[-1]))
  logitsB_Flatten = jnp.reshape(logitsB, (-1, logitsB.shape[-1]))

  # shape = (S,B,E)
  batch_Target_F = jnp.pad(batch_Input_F[1:,:,:], pad_width = ((0, 0), (0, 1), (0, 0)))
  batch_Target_B = jnp.pad(batch_Input_B[1:,:,:], pad_width = ((0, 0), (0, 1), (0, 0)))

  # shape = (E,S*B)
  batch_Target_F_Flatten = jnp.reshape(batch_Target_F, (batch_Target_F.shape[-1], -1))
  batch_Target_B_Flatten = jnp.reshape(batch_Target_B, (batch_Target_B.shape[-1], -1))

  # shape = (S*B,S*B)
  scoresF = jnp.matmul(logitsF_Flatten, batch_Target_F_Flatten)
  scoresB = jnp.matmul(logitsB_Flatten, batch_Target_B_Flatten)

  # shape = (S*B,S*B)
  softmaxF = jax.nn.log_softmax(scoresF)
  softmaxB = jax.nn.log_softmax(scoresB)

  # shape = (S*B)
  crossEntropyF = jnp.diagonal(softmaxF)*(-1)
  crossEntropyB = jnp.diagonal(softmaxB)*(-1)

  ObjF = jnp.sum(crossEntropyF) / (sequence_length * batch_size)
  ObjB = jnp.sum(crossEntropyB) / (sequence_length * batch_size)

  Obj = ObjF + ObjB

  return Obj
