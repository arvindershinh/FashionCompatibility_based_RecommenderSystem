import json
import os
import random
import sys
import pickle
from PIL import Image

import numpy as np
import pandas as pd

from numpy import asarray
import jax.numpy as jnp
import jax
from jax import jit
from jax import random


import haiku as hk

import matplotlib.pyplot as plt
import transferlearningresnet as resnetModel
import bi_lstm_fashion_layers

from outfitprocessing import wordDictionary


path = ''
param_path = os.path.join(path, 'capture/check/params.p')
params = pickle.load(open(param_path, "rb"))

print("KEYS & VALUES of MODEL PARAMS")
print(params.keys())

print(params.items())
# old Params --> dict_keys(['linear', 'lstm/linear', 'lstm_1/linear', 'mlp/~/linear_0', 'mlp_1/~/linear_0', 'visual_semantic'])




# **TEST FEATURES LOADING**


test_features_path = os.path.join(path, 'train_features.p')

test_features = pickle.load(open(test_features_path, "rb"))
image_names, image_RNNs, image_visualSemantics = test_features
len(image_names), type(image_names), image_RNNs.shape, image_visualSemantics.shape, type(image_RNNs)

image_RNNs_stack = np.vstack((image_RNNs, np.zeros(image_RNNs.shape[1])))
embedding_features = image_names, image_RNNs_stack, image_visualSemantics

outfitImages_path = os.path.join(path, 'Data/images')

# **QUERY**


query_file = os.path.join('Data/query_new.json')
query_sets = json.load(open(query_file))

input_path = query_sets['image_directory']


# **IMAGES EXTRACTION**

def embeddingGenerator(image, img_size, model, embedding_params):
    Param_RNN, Param_VisualSemantic = embedding_params
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_resized = image.resize(img_size[:-1])
    img_numpy = asarray(img_resized)
    image_feature = model.predict(np.array([img_numpy]))
    image_feature = image_feature.reshape([-1])

    image_embedding_RNN = jnp.dot(jnp.transpose(Param_RNN), image_feature)
    image_embedding_visualSemantic = jnp.dot(jnp.transpose(Param_VisualSemantic), image_feature)
    return image_embedding_RNN, image_embedding_visualSemantic


def input_images(jsonObj):
    input_path = jsonObj['image_directory']
    imgIDs = jsonObj['image_query']
    images = []
    input_image_paths = []
    for id in imgIDs:
        image_path = os.path.join(input_path, str(id) + '.jpg')
        image = Image.open(image_path)
        # print(image.format, image.size)
        images.append(image)
        input_image_paths.append(image_path)
        # print(image_path)

    return images, input_image_paths


images, input_images_paths = input_images(query_sets)

print(images)

# **CAPTION EXTRACTION**

caption_query = 'bag jacket dheheh rgrbtthth coat denim'

# **WORD DICTIONARY GENERATION**


dict_path = os.path.join(path, 'Data/final_word_dict.txt')
wordDic = wordDictionary(dict_path)

# **RESNET MODEL INITIALIZATION**


imgSize = (224, 224, 3)
modelResnet = resnetModel.getResNetModel(imgSize)


# **OUTFIT PREDICTION**

# --> PREDICTION LOGIC


def captionProcessing(caption, tokenDic):
    dict_len = len(tokenDic)

    def oneHotEncoding(tokenIds, vectorLength):
        zeroVector = [0] * vectorLength
        oneHotVectors = [zeroVector[:i] + [1] + zeroVector[i + 1:] for i in tokenIds]
        oneHotVectorsArray = np.array(oneHotVectors, dtype=jnp.float32)
        return oneHotVectorsArray

    token_ids = [tokenDic.get(word, dict_len) for word in caption.split()]
    if not (len(token_ids)):
        token_ids = [dict_len]
    oneHotVectors = oneHotEncoding(token_ids, dict_len + 1)
    oneHotEmbedding = np.sum(oneHotVectors, axis=0)
    oneHotEmbedding[-1] = 0
    return jnp.array(oneHotEmbedding, dtype=jnp.float32)


def unit(M):
    M_norm = jnp.reshape(jnp.linalg.norm(M, axis=1), (-1, 1))
    return M / M_norm


def run_RNN(rnn_cell_model, input_feed, test_feat, params_LSTM):
    logits, state = rnn_cell_model.apply(params_LSTM, input_feed, None)

    res_set = []

    while True:

        if len(res_set) >= 10:
            break

        curr_score = jnp.exp(jnp.dot(test_feat, logits.reshape(-1)))
        curr_score /= jnp.sum(curr_score)

        sorted_args = jnp.argsort(curr_score)[::-1].tolist()

        for exclusion in res_set:
            sorted_args.remove(exclusion)

        if not (len(sorted_args)):
            break

        next_image = sorted_args[0]

        if (next_image == test_feat.shape[0] - 1 or curr_score[-1] > 0.00001):
            break

        input_feed = test_feat[next_image]
        logits, state = rnn_cell_model.apply(params_LSTM, input_feed, state)
        res_set.append(next_image)

    return res_set


def remove_duplicates(seqb, seqf):
    for x in seqf:
        if x in seqb:
            seqb.remove(x)

    return seqb, seqf


def run_set_inference(imageEmbeddingRNN, test_ids, test_feat, biLSTM_params):
    params_F, params_B = biLSTM_params

    def RNN(image_embedding, initial_state):
        batch = jnp.reshape(image_embedding, (1, -1))
        core = bi_lstm_fashion_layers.LSTM_Model(False)

        if not (initial_state):
            initial_state = core.initial_state(1)

        logits, state = core(batch, initial_state)

        return logits, state

    rnn_model = hk.transform(RNN)
    loss_without_rng = hk.without_apply_rng(rnn_model)

    f_set = run_RNN(loss_without_rng, imageEmbeddingRNN[0], test_feat, params_F)
    b_set = run_RNN(loss_without_rng, imageEmbeddingRNN[0], test_feat, params_B)[::-1]

    b_set, f_set = remove_duplicates(b_set, f_set)

    outfit_components = b_set, f_set, None

    if len(imageEmbeddingRNN) >= 2:

        f_set_img_RNN_unit = unit(test_feat[f_set])
        b_set_img_RNN_unit = unit(test_feat[b_set])

        imageEmbeddingRNN_unit = imageEmbeddingRNN[1] / jnp.linalg.norm(imageEmbeddingRNN[1])

        f_cosine = jnp.dot(f_set_img_RNN_unit, imageEmbeddingRNN_unit)
        b_cosine = jnp.dot(b_set_img_RNN_unit, imageEmbeddingRNN_unit)

        f_argmax, f_max = jnp.argmax(f_cosine), jnp.max(f_cosine)
        b_argmax, b_max = jnp.argmax(b_cosine), jnp.max(b_cosine)

        if f_max >= b_max:
            outfit_components = b_set, (f_set[:f_argmax], f_set[f_argmax + 1:],), 'f'
        else:
            outfit_components = (b_set[:b_argmax], b_set[b_argmax + 1:],), f_set, 'b'

    return outfit_components


def fetch_seq(imgID_seq, image_names, dataPath):
    images = []
    image_paths = []
    for id in imgID_seq:
        outfit_id, image_id = image_names[int(id)].split('_')
        image_path = os.path.join(dataPath, str(outfit_id), str(image_id) + '.jpg')
        image = Image.open(image_path)
        # print(image.format, image.size)
        images.append(image)
        image_paths.append(image_path)
        # print(image_path)

    return images, image_paths


def nearestNeighbour_search(i, imgVS_Emnbedding, caption_embedding):
    score = jnp.dot(imgVS_Emnbedding, imgVS_Emnbedding[i] + 2.0 * caption_embedding)
    return jnp.argmax(score)


def updateOutfit_byCaption(item_seq, caption, tokenDictionary, embedding_params, img_embeddings):
    if caption != "":

        # Calculate the word embedding
        caption_OneHotEmbedding = captionProcessing(caption, tokenDictionary)
        captions_Len = jnp.sum(caption_OneHotEmbedding)

        if captions_Len:
            caption_normalize = caption_OneHotEmbedding / captions_Len
            caption_emb = jnp.dot(embedding_params, caption_normalize)
            caption_emb_norm = jnp.linalg.norm(caption_emb)
            caption_emb_unit = caption_emb / caption_emb_norm

            item_seq = [nearestNeighbour_search(i, img_embeddings, caption_emb_unit) for i in item_seq]

    return item_seq


# --> MAIN PROGRAM

#dict_keys(['linear', 'linear_1', 'linear_2', 'lstm/linear', 'lstm_1/linear', 'lstm_2/linear', 'lstm_3/linear',
# 'mlp/~/linear_0', 'mlp_1/~/linear_0', 'visual_semantic'])
#dict_keys(['linear', 'linear_1', 'linear_2', 'lstm/linear', 'lstm_1/linear', 'lstm_2/linear', 'lstm_3/linear',
#            'mlp/~/linear_0', 'mlp_1/~/linear_0', 'visual_semantic'])
def main(input_images, input_caption, tokenDic, images_embedding_db, images_path, model_resnet, model_params):
    img_embedding_params = (model_params['linear']['w'], model_params['visual_semantic']['wv'])
    semantic_embedding_params = model_params['visual_semantic']['ws']
    # params_tuple = list(model_params.items())
    # paramsF = dict([params_tuple[0], params_tuple[2], params_tuple[3]])

    # paramsB = {'lstm/linear': model_params['lstm_1/linear'], 'mlp/~/linear_0': model_params['mlp_1/~/linear_0']}
    paramsF = {'lstm/linear': model_params['lstm/linear'], 'lstm_1/linear': model_params['lstm_1/linear'],
               'linear': model_params['linear_1'], 'mlp/~/linear_0': model_params['mlp_1/~/linear_0']}
    paramsB = {'lstm/linear': model_params['lstm_2/linear'], 'lstm_1/linear': model_params['lstm_3/linear'],
               'linear': model_params['linear_2'], 'mlp/~/linear_0': model_params['mlp_1/~/linear_0']}
    #paramsB = {'lstm/linear': model_params['lstm_1/linear'], 'mlp/~/linear_0': model_params['mlp_1/~/linear_0']}

    # paramsF = {'lstm/linear': model_params['lstm/linear'], 'linear': model_params['linear_1']}
    # paramsB = {'lstm/linear': model_params['lstm_1/linear'], 'linear': model_params['linear_2']}

    embeddings = [embeddingGenerator(img, imgSize, model_resnet, img_embedding_params) for img in input_images]
    inputImageRNN = [embedding[0] for embedding in embeddings]
    inputImageVS_unit = [embedding[1] / jnp.linalg.norm(embedding[1]) for embedding in embeddings]

    image_names, image_RNNs, image_VSs = images_embedding_db
    image_VSs_norm = jnp.reshape(jnp.linalg.norm(image_VSs, axis=1), (-1, 1))
    image_VSs_unit = image_VSs / image_VSs_norm

    seqB, seqF, position = run_set_inference(inputImageRNN, image_names, image_RNNs, (paramsF, paramsB))

    if position:

        if position == 'f':
            seqF1, seqF2 = seqF
            seqB = updateOutfit_byCaption(seqB, input_caption, tokenDic, semantic_embedding_params, image_VSs_unit)
            seqF1 = updateOutfit_byCaption(seqF1, input_caption, tokenDic, semantic_embedding_params, image_VSs_unit)
            seqF2 = updateOutfit_byCaption(seqF2, input_caption, tokenDic, semantic_embedding_params, image_VSs_unit)

            outfit_B, B_paths = fetch_seq(seqB, image_names, images_path)
            outfit_F1, F1_paths = fetch_seq(seqF1, image_names, images_path)
            outfit_F2, F2_paths = fetch_seq(seqF2, image_names, images_path)
            outfit = outfit_B + [input_images[0]] + outfit_F1 + [input_images[1]] + outfit_F2
            outfit_image_paths = B_paths + F1_paths + F2_paths

        if position == 'b':
            seqB1, seqB2 = seqB
            seqB1 = updateOutfit_byCaption(seqB1, input_caption, tokenDic, semantic_embedding_params, image_VSs_unit)
            seqB2 = updateOutfit_byCaption(seqB2, input_caption, tokenDic, semantic_embedding_params, image_VSs_unit)
            seqF = updateOutfit_byCaption(seqF, input_caption, tokenDic, semantic_embedding_params, image_VSs_unit)
            # seqB1, seqB2, seqF = fun_remove_duplicate((seqB1, seqB2, seqF))
            outfit_B1, B1_paths = fetch_seq(seqB1, image_names, images_path)
            outfit_B2, B2_paths = fetch_seq(seqB2, image_names, images_path)
            outfit_F, F_paths = fetch_seq(seqF, image_names, images_path)
            outfit = outfit_B1 + [input_images[1]] + outfit_B2 + [input_images[0]] + outfit_F
            outfit_image_paths = B1_paths + B2_paths + F_paths

    else:
        seqB = updateOutfit_byCaption(seqB, input_caption, tokenDic, semantic_embedding_params, image_VSs_unit)
        seqF = updateOutfit_byCaption(seqF, input_caption, tokenDic, semantic_embedding_params, image_VSs_unit)
        outfit_B, B_paths = fetch_seq(seqB, image_names, images_path)
        outfit_F, F_paths = fetch_seq(seqF, image_names, images_path)
        outfit = outfit_B + [input_images[0]] + outfit_F
        outfit_image_paths = B_paths + F_paths

    return outfit, outfit_image_paths


# --> OUTFIT PREDICTION


outfit, outfit_paths = main(images, caption_query, wordDic, embedding_features, outfitImages_path, modelResnet, params)

new_outfits = []
import random


for item in outfit_paths:
    r = random.randint(1,6)
    a = item.split("/")[2].replace(".jpg","")
    if r!= a:
      suffix = str(random.randint(1,6)) + ".jpg"
    else:
      suffix = str(random.randint(1, 6)) + ".jpg"
    print("item : ", item, item[:-5], suffix)
    new_outfits.append(item[:-5] + suffix)

outfit_paths += new_outfits
print("Input images : ", input_images_paths )
outfit_paths = input_images_paths + list(set(outfit_paths))
print("Input + Total recommended items : ", outfit_paths)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

rows = 2
columns = 4

plt.figure(figsize=(30, 15))

for i, image in enumerate(outfit_paths):
    print("Showing : ", image, len(image))
    plt.subplot(int(len(outfit_paths) / columns + 1), columns, i + 1)
    plt.axis('off')
    if image in input_images_paths:
        plt.title("Input")
    else:
        plt.title(image.split("/")[2])
    img = mpimg.imread(image)
    plt.imshow(img)
plt.show()