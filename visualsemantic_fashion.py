

# !pip install git+https://github.com/deepmind/dm-haiku

from typing import Iterator, Tuple, Mapping

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

from jax import random

# import outfitprocessing as data

# OutfitBatchData = data.sequence_Tensor
# OutfitBatchData['outfitSequencesImage'].shape, OutfitBatchData['outfitSequencesCaption'].shape

# V = OutfitBatchData['outfitSequencesImage']
# S = OutfitBatchData['outfitSequencesCaption']

# images   = jnp.reshape(V, (-1, V.shape[-1]))
# captions = jnp.reshape(S, (-1, S.shape[-1]))

# images.shape, captions.shape

# captions_Len = jnp.reshape(jnp.sum(captions, axis=1), (-1,1))
# captions.shape, captions_Len.shape

# VisualSemanticData = {'img_batch' : images, 'caption_batch' : captions/captions_Len}

# VisualSemanticData['img_batch'].shape, VisualSemanticData['caption_batch'].shape, VisualSemanticData['img_batch'].dtype, VisualSemanticData['caption_batch'].dtype, type(VisualSemanticData['img_batch']), type(VisualSemanticData['caption_batch'])

class visual_semantic(hk.Module):

  def __init__(self, embedding_dimension, m, name=None):
    super().__init__(name=name)
    self.embedding_dimension = embedding_dimension
    self.m = m

  def __call__(self, vs):
    v = vs['img_batch']
    s = vs['caption_batch']

    i,j,k = v.shape[-1], self.embedding_dimension, s.shape[-1]

    wv_init = hk.initializers.TruncatedNormal(1. / np.sqrt(j))
    ws_init = hk.initializers.TruncatedNormal(1. / np.sqrt(j))

    wv = hk.get_parameter("wv", shape=[i, j], dtype=v.dtype, init=wv_init)
    ws = hk.get_parameter("ws", shape=[j, k], dtype=s.dtype, init=ws_init)

#  ve-->(NumberOfImages, EmbeddingSize) v-->(NumberOfImages, SizeOfImage) wv-->(SizeOfImage, EmbeddingSize)
    ve = jnp.matmul(v,wv)   
#  sT-->(SizeOfCaption, NumberOfCaptions) s-->(NumberOfCaptions, SizeOfCaption)
    sT = jnp.transpose(s)
#  se-->(EmbeddingSize, NumberOfCaptions) ws-->(EmbeddingSize, SizeOfCaption) sT-->(SizeOfCaption, NumberOfCaptions)    
    se = jnp.matmul(ws, sT)

    ve_m = jnp.reshape(jnp.linalg.norm(ve, axis=1), (-1,1))
    se_m = jnp.reshape(jnp.linalg.norm(se, axis=0), (1,-1))

    ve_u = ve/ve_m
    se_u = se/se_m

    v_CosMatrix_s = jnp.matmul(ve_u, se_u)
    vN, sN = v_CosMatrix_s.shape
    
    v_CosDiagonal_s = jnp.diagonal(v_CosMatrix_s)

    v_VARs = self.m+jnp.subtract(v_CosMatrix_s, jnp.reshape(v_CosDiagonal_s, (-1,1)))
    s_VARv = self.m+jnp.subtract(v_CosMatrix_s, jnp.reshape(v_CosDiagonal_s, (1,-1)))

    zeroMatrix = jnp.zeros_like(v_CosMatrix_s)

    v_VARs_max = jnp.maximum(v_VARs,zeroMatrix)
    s_VARv_max = jnp.maximum(s_VARv,zeroMatrix)

    v_VARs_final = jnp.subtract(v_VARs_max, jnp.diag(jnp.diag(v_VARs_max)))
    s_VARv_final = jnp.subtract(s_VARv_max, jnp.diag(jnp.diag(s_VARv_max)))

    return (jnp.sum(v_VARs_final)+jnp.sum(s_VARv_final))/(vN*sN)

def visual_semantic_fn(vs) -> jnp.ndarray:
  embeddingSize = 512
  m = .2
  model = visual_semantic(embeddingSize, m)
  return model(vs)
