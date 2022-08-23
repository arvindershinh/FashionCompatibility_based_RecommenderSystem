import os
import matplotlib.image as mpimg
import numpy as np
import streamlit as st
from PIL import Image
from load_css import local_css
from annotated_text import annotated_text

# Importing Model Python file
from dummyModel import *

uploaded_files = ""
input_image_list =[]
output_image_list =[]

local_css("css/style.css")
title="<div><span><h1 class='title bold' >Fashion Recommender System</h1></span></div>"
menu_items = "<div class='menu'> <span class='menu'><span class='highlight blue bold'>Men</span><span class='highlight blue bold'>Women</span><span class='highlight blue bold'>  Kids </span><span class='highlight blue bold'>Accessories </span><span class='highlight blue bold'>Jewellery</span><span class='highlight blue bold'>Footwear </span></span></div>"
display_folder="displayImages"
input_folder = "inputImage"
output_folder = "outputImage"

st.markdown(title, unsafe_allow_html=True)
st.markdown(menu_items, unsafe_allow_html=True)

input_text = st.text_input('Fashion Preference', value="", max_chars=None, key="inputText", type="default",placeholder="Enter your Fashion Preference...........")
uploaded_files = st.file_uploader("Choose an Image", accept_multiple_files=True,key="imageFiles")
recommedation_button=st.button('Get Recommendation')
reset_button=st.button(' Reset Data ')

col1, col2,col3 = st.columns([1,1,1])


# Empty input folder
def empty_input_folder():
    for filename in os.listdir(input_folder):
        os.remove(os.path.join(input_folder, filename))

# Empty output folder
def empty_output_folder():
    for filename in os.listdir(output_folder):
        os.remove(os.path.join(output_folder, filename))
#Reset Data
def reset_data():
    input_text = ""
    uploaded_files = []
    #empty_output_folder()
    empty_input_folder()
    for session_key in st.session_state.keys():
        print("session_key::")
        print(session_key)
        #st.session_state[session_key] = ""
        #st.session_state["inputText"] = ''
        #if(key== 'inputText') :
            #st.session_state.inputText = ""
        #del st.session_state[key]

# Function to Display Recommended Images
def display_output_image(image_list):


    t = "<div><span class='recom bold'>Our Recommendations :</span></div>"


    #for filename in os.listdir(folder):
    print("Size of output Image List")
    print(len(image_list))
    with col3:
        st.markdown(t, unsafe_allow_html=True)
        st.markdown("----------------------------------------")
    for filename in image_list:
        with col3:
            st.image(filename)

# Function to Display Images in left panel
def display_image():

    for filename in os.listdir(display_folder):
        img = mpimg.imread(os.path.join(display_folder, filename))
        with col1:
            st.image(img)

# Function to Read and multiple Images
def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image

# display Input Image
def load_container():
    t1 = "<div><span class='recom bold'>Your selections :</span></div>"

    with col2:

        if len(uploaded_files)>0 or input_text != "":
            st.markdown(t1,unsafe_allow_html=True)
            st.markdown("-----------------------------------")
            st.markdown("")
            annotated_text(("  ",input_text, "yelllow","blue"))
            st.markdown("")

        imageContainer = st.container()
        empty_input_folder()
        if uploaded_files is not None:
             for uploaded_file in uploaded_files:
                 save_img=Image.open(uploaded_file)
                 save_img.save(os.path.join(input_folder, uploaded_file.name), "")
                 input_image_list.append(save_img)
                 img = load_image(uploaded_file)
                 imageContainer.image(img)
        print("Size of input Image List after upload:::")
        print(len(input_image_list))

def get_recommendation():
    with col2:
        if reset_button:
            image_list = []
            reset_data()

    with col2:
        if recommedation_button:
            image_list = []
            # empty_output_folder()
            for filename in os.listdir(input_folder):
                img = Image.open(os.path.join(input_folder, filename))
                image_list.append(img)
            print("get_recommendation-->output_image_list")
            print(len(image_list))
            new_image_list = recommendItem(image_list, input_text)
            print(len(new_image_list))
            display_output_image(new_image_list)


get_recommendation()
display_image()
load_container()





