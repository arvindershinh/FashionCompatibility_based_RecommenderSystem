

import json
import os
import random
import sys
import pickle

import numpy as np
import pandas as pd

from PIL import Image
from numpy import asarray
import jax.numpy as jnp
import jax
from jax import jit
from jax import random
import numba as nb

# from google.colab import drive
# drive.mount('/content/gdrive')

# sys.path.insert(0,'/content/gdrive/MyDrive/Colab Notebooks/Capstone Project/Dependencies')

import transferlearningresnet as resnetModel
import pickle
import multiprocessing as mp
from datetime import datetime
"""TEST DATA LOADING"""

# path = 'gdrive/MyDrive/Colab Notebooks/Capstone Project/Polyvore_Data/'
path = "Data/"
json_file = os.path.join(path, 'test_no_dup.json')

all_sets = json.load(open(json_file))

"""PARAMETERS LOADING"""

# param_path = '/content/gdrive/MyDrive/Colab Notebooks/Capstone Project/params.p'
param_path = "params.p"
params = pickle.load(open(param_path, "rb"))
params.keys()

#print(params['linear']['w'].shape, params['visual_semantic']['wv'].shape)

"""EMBEDDING GENERATION"""

imgSize = (224, 224, 3)
model_resnet = resnetModel.getResNetModel(imgSize)
testData_path = 'Data/images'
embedding_params = (params['linear']['w'], params['visual_semantic']['wv'])


def embeddingGenerator(image):
    Param_RNN, Param_VisualSemantic = embedding_params
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_resized = image.resize(imgSize[:-1])
    img_numpy = asarray(img_resized)
    image_feature = model_resnet.predict(np.array([img_numpy]))
    image_feature = image_feature.reshape([-1])

    image_embedding_RNN = jnp.dot(jnp.transpose(Param_RNN), image_feature)
    image_embedding_visualSemantic = jnp.dot(jnp.transpose(Param_VisualSemantic), image_feature)

    return image_embedding_RNN, image_embedding_visualSemantic


def parallel_outfits_process(outfits):
    image_names = []
    image_RNNs = []
    image_visualSemantics = []

    for outfit in outfits:
        print("outfit id : ", outfit['set_id'])
        set_id = outfit['set_id']
        outfit_path = os.path.join(testData_path, str(set_id))
        if os.path.exists(outfit_path):
            for image_info in outfit['items']:
                image_name = set_id + "_" + str(image_info["index"])
                image_path = os.path.join(outfit_path, str(image_info['index']) + '.jpg')
                print(image_path)
                image = Image.open(image_path)
                image_RNN, image_visualSemantic = embeddingGenerator(image)
                image_names.append(image_name)
                image_RNNs.append(image_RNN)
                image_visualSemantics.append(image_visualSemantic)

    return (image_names, np.array(image_RNNs), np.array(image_visualSemantics))


def imageEmbedding(outfits):
    print("Number of processors: ", mp.cpu_count())
    chunksize = 100
    with mp.Pool(processes=mp.cpu_count()) as pool:
        result = pool.map(parallel_outfits_process, [outfits], chunksize)

    return result

if __name__ == '__main__':

    now = datetime.now()
    test_features = imageEmbedding(all_sets)
    print(test_features)
    a, b, c = test_features[0]

    print("test features : ", len(a), type(a), b.shape, c.shape)



    test_features_path = 'test_features.p'

    pickle.dump(test_features[0], open(test_features_path, "wb"))
    print("test_features.p is ready!")
    print("Started at : ", now)
    print("Completed at : ", datetime.now())
