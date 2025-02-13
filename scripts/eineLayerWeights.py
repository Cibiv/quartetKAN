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

#modell laden:
model = keras.saving.load_model('../models/F-zoneNN_20250212_072800_003-0.667-0.668')
#anschauen was es für layer gibt: 
#model.summary()
#auf den i-ten layer des models mit models.layers[i] zugreifen
#hidden layer:
#print(model.layers[0])
#<keras.src.saving.legacy.saved_model.load.DenseKAN object at 0x7f4d8748d850>

#output layer:
#print(model.layers[1])

#für dense layers erhält man die weights mit model.layers[i].weights
#print(model.layers[1].weights)


#wie erhalte ich die weights und aktivierungsfunktionen von den hidden layers?
print(model.layers[0].weights)

