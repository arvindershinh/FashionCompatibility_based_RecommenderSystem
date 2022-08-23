import os
import matplotlib.image as mpimg
import numpy as np
import streamlit as st
from PIL import Image
from load_css import local_css
input_text = ""
uploaded_files = ""
model_status = ""

local_css("C:\Deepti\IISC\CapstoneProject\ModelWebApp\css\style.css")

title="<div><span><h1 class='title bold' >Fashion Recommender System</h1></span></div>"
st.markdown(title, unsafe_allow_html=True)
t = "<div class='menu'> <span class='menu'><span class='highlight blue bold'>Men</span><span class='highlight blue bold'>Women</span><span class='highlight blue bold'>  Kids </span><span class='highlight blue bold'>Accessories </span><span class='highlight blue bold'>Jewellery</span><span class='highlight blue bold'>Footwear </span><span class='highlight blue bold'>Beauty</span><span class='highlight blue bold'>Home Decor</span></span></div>"
st.markdown(t, unsafe_allow_html=True)
input_text = st.text_input('Fashion Preference', '')
uploaded_files = st.file_uploader("Choose an Image", accept_multiple_files=True)
col1, col2,col3 = st.columns([1,1,1])

# Function to Display Images in left column
def display_image():
    folder="C:\Deepti\IISC\CapstoneProject\ModelWebApp\displayImages"
    images = []
    with col1:
        st.markdown("----------------------------")
    for filename in os.listdir(folder):
        img = mpimg.imread(os.path.join(folder, filename))
        with col1:
            st.image(img)

        if img is not None:
            images.append(img)
    return images

# Function to Display Recommended Images
def display_output_image():
    folder = "C:\Deepti\IISC\CapstoneProject\ModelWebApp\outputImage"
    t = "<div><span class='recom bold'>Recommendations :</span></div>"
    with col3:
        st.markdown("----------------------------")
        st.markdown(t, unsafe_allow_html=True)
        st.markdown("----------------------------")
    images = []

    for filename in os.listdir(folder):
        img = mpimg.imread(os.path.join(folder, filename))
        with col3:
            st.image(img, width=80)

        if img is not None:
            st.write('----------------------------------------------------------------------------------')
            images.append(img)
    return images

# Function to Read and multiple Images
def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image

def reset_data():
    input_text = ""
    uploaded_files = ""
    model_status = ""

#def save_input_image(img):
    #im = Image.open(img)

#def empty_input_image():
    #EMPTY

def load_output_container():
    t1 = "<div><span class='recom bold'>Your selections :</span></div>"
    #cols = imageContainer.columns(3)
    with col2:
        st.markdown("----------------------------")

        st.markdown(t1,unsafe_allow_html=True)
        st.markdown("----------------------------")
        imageContainer = st.container()
        if uploaded_files is not None:
             for uploaded_file in uploaded_files:
                  img = load_image(uploaded_file)
                  imageContainer.image(img,width=70)


def get_recommendation():

    if st.button('Get Recommendation'):
        # model_status= recom_model(input_text,uploaded_files)
        model_status = "SUCCESS"
        if model_status=="SUCCESS":
            display_output_image()
    else:
        st.write('')

get_recommendation()
display_image()
load_output_container()


