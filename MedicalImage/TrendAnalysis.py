import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,models

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import os
import random
import shutil

model_builder = keras.applications.xception.Xception
img_size = (224, 224)
preprocess_input = keras.applications.xception.preprocess_input
decode_predictions = keras.applications.xception.decode_predictions

#The last convolutional layer name in the architecture
last_conv_layer_name = "conv2d_4"

# The local path to our target image and img showing test
img_path = "/home/akiba/newcomer/assignment/2_MedicalImageClassification/Dataset/FN/67.png"
display(Image(img_path))

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    #array = array.astype("float")/255.
    array = np.expand_dims(array, axis=0)
    print(array.shape)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    print(grads.shape)
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    print(pooled_grads.shape)

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    print(last_conv_layer_output.shape)
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)


    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    #display(Image(cam_path))

# Make model
model = models.load_model('/home/akiba/newcomer/assignment/2_MedicalImageClassification/model.h5')
print(model.summary())

# Remove last layer's softmax
model.layers[-1].activation = None

# Path to save GradCAM
cam_path = "/home/akiba/newcomer/assignment/2_MedicalImageClassification/Dataset/GradCAM"
shutil.rmtree(cam_path)
os.mkdir(cam_path)

# Path for images
FNlist = os.listdir("/home/akiba/newcomer/assignment/2_MedicalImageClassification/Dataset/FN")
FPlist = os.listdir("/home/akiba/newcomer/assignment/2_MedicalImageClassification/Dataset/FP")
TNlist = os.listdir("/home/akiba/newcomer/assignment/2_MedicalImageClassification/Dataset/TN")
TPlist = os.listdir("/home/akiba/newcomer/assignment/2_MedicalImageClassification/Dataset/TP")

FNrandom = random.sample(FNlist, 10)
FPrandom = random.sample(FPlist, 10)
TNrandom = random.sample(TNlist, 10)
TPrandom = random.sample(TPlist, 10)

# Generate GradCAM
for i in FNrandom:
    #prepare image
    img_path = "/home/akiba/newcomer/assignment/2_MedicalImageClassification/Dataset/FN/"+i
    img_array = preprocess_input(get_img_array(img_path, size=img_size))

    #generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    save_and_display_gradcam(img_path, heatmap,cam_path+"/FN_"+i)

for i in FPrandom:
    #prepare image
    img_path = "/home/akiba/newcomer/assignment/2_MedicalImageClassification/Dataset/FP/"+i
    img_array = preprocess_input(get_img_array(img_path, size=img_size))

    #generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    save_and_display_gradcam(img_path, heatmap,cam_path+"/FP_"+i)

for i in TNrandom:
    #prepare image
    img_path = "/home/akiba/newcomer/assignment/2_MedicalImageClassification/Dataset/TN/"+i
    img_array = preprocess_input(get_img_array(img_path, size=img_size))

    #generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    save_and_display_gradcam(img_path, heatmap,cam_path+"/TN_"+i)

for i in TPrandom:
    #prepare image
    img_path = "/home/akiba/newcomer/assignment/2_MedicalImageClassification/Dataset/TP/"+i
    img_array = preprocess_input(get_img_array(img_path, size=img_size))

    #generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    save_and_display_gradcam(img_path, heatmap,cam_path+"/TP_"+i)