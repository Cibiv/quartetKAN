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
#print(model.layers[2].weights)
#[<tf.Variable 'dense/kernel:0' shape=(2, 1) dtype=float32, numpy=
#array([[ 2.5388613e+00],
 #      [-1.0384964e-03]], dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(1,) dtype=float32, numpy=array([-0.52242494], dtype=float32)>]
#in meiner letzten laydr werden also die zwei nodes, die von der zweiten hidden layer übergeben werden mit 2.5 und -1.0 multipliziert, 
#dann -0.5224 dazuaddiert und dann durch die sigmoid funktion gegeben für den output

#wie erhalte ich die weights und aktivierungsfunktionen von den hidden layers?
#inspiration bei pykan für die symbolic function
#print(model.layers[1].weights)
"""[<tf.Variable 'dense_kan_1/spline_kernel:0' shape=(4, 8, 2) dtype=float32, numpy=
array([[[ 1.2408248 ,  0.6007983 ],
        [ 0.92225593,  0.34695032],
        [-1.1237121 , -0.72650015],
        [-3.1709268 , -1.743789  ],
        [-5.539622  , -4.0989113 ],
        [-0.31851938, -0.7420227 ],
        [ 0.0976807 , -0.7406196 ],
        [ 1.2272903 ,  0.7075231 ]],

       [[-0.4713877 ,  2.1175296 ],
        [ 0.36328027,  1.3970504 ],
        [ 1.0262586 , -0.03407883],
        [ 1.2201215 , -0.48507842],
        [ 1.6582799 , -1.0502591 ],
        [-0.31799054,  0.5480379 ],
        [ 0.14940225, -0.4797849 ],
        [ 0.3275606 , -1.872456  ]],

       [[-0.68110937, -0.8740852 ],
        [-1.1184362 , -0.8503553 ],
        [-1.9171951 , -1.3097199 ],
        [-1.5326462 , -0.63965917],
        [-0.87781453, -0.65528256],
        [ 1.5727004 ,  1.0952253 ],
        [ 1.7911612 , -0.2908777 ],
        [ 4.7530837 ,  2.543891  ]],

       [[ 1.394718  ,  1.6258446 ],
        [ 0.26839685,  0.03791697],
        [-0.89680916, -1.5997666 ],
        [-0.6970364 , -0.7397104 ],
        [-1.3714889 , -1.2127262 ],
        [ 2.8874383 ,  2.057014  ],
        [ 1.236367  ,  0.7994961 ],
        [ 0.4134899 ,  1.1370059 ]]], dtype=float32)>, <tf.Variable 'dense_kan_1/scale_factor:0' shape=(4, 2) dtype=float32, numpy=
array([[-0.6611737 ,  0.19429201],
       [-2.8217506 , -0.2748935 ],
       [ 1.6778848 , -0.53418183],
       [ 1.3091022 , -0.24741721]], dtype=float32)>, <tf.Variable 'dense_kan_1/bias:0' shape=(2,) dtype=float32, numpy=array([-0.55845827,  0.5807731 ], dtype=float32)>, <tf.Variable 'dense_kan_1/spline_grid:0' shape=(4, 12) dtype=float32, numpy=
array([[-2.2       , -1.8000001 , -1.4000001 , -1.        , -0.6       ,
        -0.20000005,  0.20000005,  0.5999999 ,  1.        ,  1.4000001 ,
         1.8       ,  2.2       ],
       [-2.2       , -1.8000001 , -1.4000001 , -1.        , -0.6       ,
        -0.20000005,  0.20000005,  0.5999999 ,  1.        ,  1.4000001 ,
         1.8       ,  2.2       ],
       [-2.2       , -1.8000001 , -1.4000001 , -1.        , -0.6       ,
        -0.20000005,  0.20000005,  0.5999999 ,  1.        ,  1.4000001 ,
         1.8       ,  2.2       ],
       [-2.2       , -1.8000001 , -1.4000001 , -1.        , -0.6       ,
        -0.20000005,  0.20000005,  0.5999999 ,  1.        ,  1.4000001 ,
         1.8       ,  2.2       ]], dtype=float32)>]

"""
print(model.layers[0].weights)



