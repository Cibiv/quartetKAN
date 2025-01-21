import numpy as np
import pandas as pd
import tensorflow.keras as keras
from keras import backend as K
import tensorflow as tf


import argparse
import logging
import os

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from pylab import *

matplotlib.use('Agg')

# pass arguments with flags
parser = argparse.ArgumentParser()

parser.add_argument("-m", "--model", help = "Model to test.")
parser.add_argument("-t", "--test", default = '../data/processed/zone/test/1000bp', help = "Path to test directory")
parser.add_argument("-l", "--seqlen", default = '1000bp', help = "Sequence Length")
parser.add_argument("-o", "--offset", type = int, default = 4, help = "Index of first feature column.")
parser.add_argument("-n", "--no_feat", type = int, default = 15, help = "Number of feature columns.")
args = vars(parser.parse_args())

model = keras.saving.load_model(args['model'])

#print the model summary
model.summary()

#iterate through layers and extract weights and biases
#for layer in model.layers:
 #   print(f"Layer: {layer.name}")
  #  if hasattr(layer, 'get_weights'):
   #     weights = layer.get_weights()
    #    if len(weights) > 0:
     #       print(f"  Weights: {weights[0]}")  #weight matrix
      #      print(f"  Biases: {weights[1]}")   #bias vector


#print activation functions for each layer
#for layer in model.layers:
 #   if hasattr(layer, 'activation'):
  #      print(f"Layer: {layer.name}, Activation: {layer.activation.__name__}")

#######
##output: Layer: dense, Activation: sigmoid
######



#for layer in model.layers:
 #       if hasattr(layer, 'get_weights'):
  #          print(f"  Weights and parameters: {layer.get_weights()}")
#####
#output: einfach suuuuper viele Zahlen, gar nicht übersichtlich :( 
#####
#for layer in model.layers:
 #   print(f"Layer name: {layer.name}, Layer type: {type(layer)}")
  #  if hasattr(layer, 'activation'):
   #     print(f"Activation: {layer.activation}")

#for layer in model.layers:
 #   if hasattr(layer, 'weights'):
  #      weights = layer.get_weights()
   #     print(f"Weights for layer {layer.name}: {weights}")


#klappt nicht 
#model.plot()
from keras.utils import plot_model
from tensorflow.keras.utils import plot_model
#tf.keras.utils.plot_model()
#plot_model(model=model, show_shapes=True)

#Hiermit habe ich ein kleines png bild von meinem model erstellt, es enthält aber wirklich nicht viel. nur pfeile und wenig infos (DENSEKAN oder DENSE layer)
#plot_model(model, to_file='../results/model.png')

###
#https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model?hl=en 
#hier gibt es mehr infos was man alles anzeigen lassen kann 
####

plot_model(model, to_file='../results/model.png', show_layer_activations=True, show_trainable=True, show_shapes=True, show_dtype=True, show_layer_names=True, expand_nested=True)
#show_layer_activations=False,
 #   show_trainable=False,

#    show_shapes=False,
 #   show_dtype=False,
  #  show_layer_names=False,
   # rankdir='TB',
#    expand_nested=False,
 #   dpi=200,
  #  show_layer_activations=False,
   # show_trainable=False,


########################

#import sympy as sp
import tf2onnx
import onnx
import onnx2pytorch import ConvertModel

ibbx_model, _ = t2onnx.convert.from_keras(model)
pytorch_model = ConvertModel(onnx_model)
pytorch_model



#funktioniert nicht:
#model.symbolic_formula()[0][0]
#funktioniert nicht:
#print(model.plot())
#funktioniert nicht:
#print(model.symbolic_formula()[0][0])





