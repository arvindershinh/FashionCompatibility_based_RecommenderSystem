import os
import matplotlib.image as mpimg
from recom import get_recommendations

def recommendItem(input_image_list, caption):

    print("-----------------recommendItem called---------------")
    output_image_list = get_recommendations(input_image_list, caption)
    return output_image_list