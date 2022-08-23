
# ! pip install -r requirements.txt
# ! pip install transformers
# ! pip install sentence_transformers

# from google.colab import drive
# drive.mount('/content/drive')

# Data manipulation libraries
import pandas as pd
import numpy as np
import itertools
import json
import requests
# Generic utility libraries
import sys
import time
import textwrap
import os
# Model like libraries
from scipy.spatial import distance
# NLP libraries
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import torch
from tqdm.notebook import tqdm
# Image related libraries
from PIL import Image
import cv2
# JAX related libraries
from numpy import asarray
import jax.numpy as jnp
import jax
from jax import jit
from jax import random
# Custom code for image embeddings using RESNET
import transferlearningresnet as resnetModel

# sequence_path = "drive/MyDrive/Image_embeddings_FR/"
sequence_path = "Data/"

imvect = np.load(os.path.join(sequence_path, 'outfitSequencesImage.npy'), allow_pickle=True)
print("Training image embeddings size:" ,imvect.shape ,"\n")

word_vect = np.load(os.path.join(sequence_path, 'outfitSequencesCaption.npy'), allow_pickle=True)
print("One-hot encoded embeddings size:" ,word_vect.shape ,"\n" )

train_im = imvect[:14000 ,: ,:]
test_im = imvect[14000: ,: ,:]

def cosine_distances(x1 ,x2):
  if(x1.shape[:1 ]==x2.shape[-1:]):
    if(x2.shape[1 ]==1):
      return distance.cosine(x1 ,x2)
    if(len(x2.shape ) >1):
      d = np.zeros(np.array(x2.shape[:-1]))
      if len(d.shape )==2:
        for i in range(x2.shape[0]):
          for j in range(x2.shape[1]):
            d[i ,j] = distance.cosine(x1 ,x2[i ,j ,:])
        return d
      if len(d.shape )==1:
        for i in range(x2.shape[0]):
          d[i] = distance.cosine(x1 ,x2[i ,:])
        return d

def recommend_matching(im1):
  a = im1.reshape(-1 ,1)
  matout = cosine_distances(a, train_im)
  minimum = np.min(matout)
  matching_index = np.where(matout == minimum)
  
  # Remove the exact matching image from the outfit
  other_image_index_outfit = list(range(train_im[matching_index[0][0] ,: ,:].shape[0]))
  other_image_index_outfit.remove(matching_index[1][0])

  recommended_images = []
  # let 'b' be the most matching embedding for a given image
  # we want to recommend something where 'b' is a part of outfit and 'b' is not a recommended object directly
  for i in range(3):
    b2 = train_im[matching_index[0][0] ,other_image_index_outfit[i] ,:].reshape(-1 ,1)
    rec_out = cosine_distances(b2, train_im)
    indexes_sort = np.argsort(rec_out.ravel())[:3]
    rec_image_index = indexes_sort // 8 , indexes_sort % 8 
    for j in range(len(rec_image_index[0])):
      recommended_images.append((rec_image_index[0][j], rec_image_index[1][j]))
  
  return recommended_images

outfit_folders = pd.read_csv("Data/image_folders.csv" ,index_col=["Index"])
def create_recommendation_folder_file_map(match_output):
  outfit_index = [x[0] for x in match_output]
  outfit_folder = [outfit_folders.iloc[x ,:][0] for x in outfit_index]
  image_index = [x[1] + 1 for x in match_output]
  image_name = [str(x) + ".jpg" for x in image_index]
  folder_file = pd.DataFrame({'Folder': outfit_folder, 'Image': image_name})
  return folder_file

rs_output = recommend_matching(test_im[280 ,0 ,:])

"""Mapping recommendation index to folders and images"""

print(create_recommendation_folder_file_map(rs_output))

# print(rs_output2)

rs_output3 = recommend_matching(test_im[1957 ,1 ,:])

print(test_im[1957 ,1 ,:].shape)

print(rs_output3)

imgSize = (224 ,224 ,3)
model_resnet = resnetModel.getResNetModel(imgSize)

image = cv2.imread("inputs/pink.jpg")
img_resized = cv2.resize(image, (224, 224), interpolation = cv2.INTER_NEAREST)
img_numpy =  asarray(img_resized)
image_feature = model_resnet.predict(np.array([img_numpy]))

rs_output4 = recommend_matching(image_feature)

output = create_recommendation_folder_file_map(rs_output4)
print(output)

import matplotlib.pyplot as plt
rows = 2
columns = 4

plt.figure(figsize=(30, 15))



for i, image in output.iterrows():
    print("image : ", image, " i : ", i)
    plt.subplot(int(len(rs_output4) / columns + 1), columns, i + 1)
    plt.axis('off')
    img = Image.open("Data/images/" +str(image['Folder']) +"/" +str(image["Image"]))
    plt.imshow(img)
img = Image.open("inputs/pink.jpg")
plt.imshow(img)
plt.title("Input")
plt.show()

"""BERT Embeddings"""

#path = 'drive/MyDrive/Colab Notebooks/Capstone Project/Outfit_Processing/Data/'
path = 'Data/'
json_file = os.path.join(path, 'outfit.json')
dictionary_file = os.path.join(path, 'final_word_dict.txt')
import json
all_sets = json.load(open(json_file))
descriptions = [[y['name'] for y in x['items']] for x in all_sets]

#print(all_sets[0])

descriptions_list = [item for sublist in descriptions for item in sublist]
description_items = pd.DataFrame(descriptions_list)
description_items.columns =['descriptions']

import pickle
filename = os.path.join('Data/descr_embed_train_data.pickle')
with open(filename, 'rb') as handle:
  emb = pickle.load(handle)

print(emb[17315].shape)

"""Output Set 1"""
print("output : ",rs_output)
# Test image folder
print(outfit_folders.iloc[14000+280,:][0])

# Recommendation 1
print(outfit_folders.iloc[rs_output[0][0],:][0])

# Recommendation 2
print(outfit_folders.iloc[rs_output[5][0],:][0])

"""Output Set 2"""

# Test image folder  - black shoes image
print(outfit_folders.iloc[14000+1530,:][0])

# print(outfit_folders.iloc[rs_output2[5][0],:][0]) # - black purse
#
# print(outfit_folders.iloc[rs_output2[6][0],:][0]) # - ring with a red stone
#
# print(outfit_folders.iloc[rs_output2[1][0],:][0]) # - black one piece

"""Output Set 3"""

print(outfit_folders.iloc[14000+1973,:][0])

print(outfit_folders.iloc[rs_output3[1][0],:][0])  # Red full sleeves t-shirt

print(outfit_folders.iloc[rs_output3[8][0],:][0]) # High heels shoes - English and black colored

print(outfit_folders.iloc[rs_output3[6][0],:][0]) # high heels with golden and blue shoes

"""Output Set 4"""

print(outfit_folders.iloc[rs_output4[4][0],:][0])
