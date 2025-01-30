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

#matplotlib.use('Agg')

# pass arguments with flags
#parser = argparse.ArgumentParser()

#parser.add_argument("-m", "--model", help = "Model to test.")
#parser.add_argument("-t", "--test", default = '../data/processed/zone/test/1000bp', help = "Path to test directory")
#parser.add_argument("-l", "--seqlen", default = '1000bp', help = "Sequence Length")
#parser.add_argument("-o", "--offset", type = int, default = 4, help = "Index of first feature column.")
#parser.add_argument("-n", "--no_feat", type = int, default = 15, help = "Number of feature columns.")
#args = vars(parser.parse_args())

#model = keras.saving.load_model(args['model'])

#print the model summary
#model.summary()

#modell laden:
model = keras.saving.load_model('../models/F-zoneNN_20250116_140646_020-0.819-0.820')
#anschauen was es für layer gibt: [15,4,2,1],index: 0 -> 4,1 -> 2,2->1 also 4 und 2 sind DenseKAN und 1 ist normal, 15 ist die Größe vom input
model.summary()
#auf den i-ten layer des models mit models.layers[i] zugreifen
#print(model.layers[0])
#<keras.src.saving.legacy.saved_model.load.DenseKAN object at 0x7f8efa866350>
#print(model.layers[1])
#<keras.src.saving.legacy.saved_model.load.DenseKAN object at 0x7f8efa7dfd10>
#print(model.layers[2])
#<keras.src.layers.core.dense.Dense object at 0x7f8efa202f50>

#für dense layers erhält man die weights mit model.layers[i].weights
print(model.layers[2].weights)






