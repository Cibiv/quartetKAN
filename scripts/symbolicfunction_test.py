import numpy as np
import pandas as pd
import tensorflow.keras as keras
from keras import backend as K

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
for layer in model.layers:
    if hasattr(layer, 'activation'):
        print(f"Layer: {layer.name}, Activation: {layer.activation.__name__}")

#######
##output: Layer: dense, Activation: sigmoid
######


#check for custom layers or spline-related components
for layer in model.layers:
        if hasattr(layer, 'get_weights'):
            print(f"  Weights and parameters: {layer.get_weights()}")






#model.plot()
#model.symbolic_formula()[0][0]

#print(model.plot())
#print(model.symbolic_formula()[0][0])




# with a Sequential model
#get_3rd_layer_output = K.function([model.layers[0].input],
 #                                 [model.layers[1].output])
#layer_output = get_3rd_layer_output([1])[0]
