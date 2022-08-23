

# !pip install git+https://github.com/deepmind/dm-haiku

# import outfitprocessing as data
import bi_lstm_fashion as bi_lstm
import visualsemantic_fashion as VisualSemantic

from typing import Iterator, Mapping

import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp

# OutfitBatchData = data.sequence_Tensor
# OutfitBatchData['outfitSequencesImage'].shape, OutfitBatchData['outfitSequencesCaption'].shape

def total_loss(batch):
  image_Tensor, caption_Tensor = batch['outfitSequencesImage'], batch['outfitSequencesCaption']

  B, S, E = image_Tensor.shape
  image_Tensor_biLSTM = jnp.reshape(image_Tensor, (S, B, E))

  def visualSemanticInput(x):
    img_Tensor, captn_Tensor = x
    images   = jnp.reshape(img_Tensor, (-1, img_Tensor.shape[-1]))
    captions = jnp.reshape(captn_Tensor, (-1, captn_Tensor.shape[-1]))
    captions_Len = jnp.reshape(jnp.sum(captions, axis=1), (-1,1))
    return {'img_batch' : images, 'caption_batch' : captions/captions_Len}

  VisualSemanticData = visualSemanticInput((image_Tensor, caption_Tensor))

  objF_objB_objVS = bi_lstm.sequence_loss(image_Tensor_biLSTM)+VisualSemantic.visual_semantic_fn(VisualSemanticData)

  return objF_objB_objVS
