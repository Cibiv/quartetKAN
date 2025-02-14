import numpy as np
from scipy.interpolate import CubicSpline
from numpy.polynomial.polynomial import Polynomial

#Beispiel-Frequenzvektor 
freq_vector = np.array([[0.079, 0.019, 0.135, 0.019, 0.031, 0.04, 0.129, 0.04, 
                         0.02, 0.142, 0.041, 0.032, 0.19, 0.038, 0.045]])

#Weights aus dem Modell mit nur einem node in der einzigen hidden layer
spline_kernel = np.array([
    [[-2.691, -0.0316, 3.513, -0.2019, 0.3156, -0.2999, -0.6609, 0.5346]],
    [[-0.090, -0.123, 0.283, -0.151, 0.131, -1.015, -1.785, 0.154]],
    [[0.0115, 0.0219, -0.569, -0.169, 0.243, -1.203, -4.900, -0.037]],
    [[0.0203, 0.0389, 3.435, 0.147, -0.228, 0.410, 0.789, 0.018]],
    [[0.0459, -0.157, -4.093, 0.527, 0.0085, -1.139, 24.21, -0.035]],
    [[-0.119, 0.156, -0.0056, 0.422, -0.330, 0.332, 1.115, 0.093]],
    [[0.0753, -0.0571, -3.880, 0.0728, 0.299, -1.738, -0.604, 0.019]],
    [[-0.0427, 0.0121, -2.959, 0.281, -0.147, -0.799, 16.41, -0.098]],
    [[-0.0415, 0.124, 3.782, 0.136, -0.229, 0.453, 0.655, -0.068]],
    [[-0.0913, -0.0701, -0.519, -0.113, 0.196, -1.035, -5.378, 0.115]],
    [[-0.0113, -0.0621, 4.267, -0.192, -0.085, 0.124, -21.59, -0.051]],
    [[-0.0413, -0.119, 4.078, -0.0719, 0.0246, 0.149, -19.30, -0.194]],
    [[-0.0371, 0.0622, 3.445, -0.0837, -0.0356, 0.0216, -17.05, -0.095]],
    [[-0.0343, -0.0483, -3.376, 0.346, -0.118, -0.800, 17.48, -0.021]],
    [[-0.111, -0.0991, 3.398, 0.109, -0.256, 0.453, 0.162, -0.108]]
])

scale_factor = np.array([
    [0.5208], [1.978], [1.3029], [2.4282], [-1.5205], [-1.7595], [-1.6266],
    [-3.1435], [2.2859], [1.5013], [2.2842], [2.6324], [3.1901], [-2.8730], [-3.1667]
])

bias = np.array([0.0403])

#Spline-Grid

spline_grid = np.array([-2.2, -1.8000001, -1.4000001, -1.0, -0.6, 
                        -0.20000005, 0.20000005, 0.5999999, 1.0, 
                        1.4000001, 1.8, 2.2])
                        

#print(f"spline_grid[:12]: {spline_grid[:12]}")
#print(f"Anzahl der Punkte in spline_grid[:12]: {len(spline_grid[:12])}")

#spline = CubicSpline(spline_grid[:12], spline_kernel[i, :12, 0])

#print(f"spline_kernel[{i}, :, 0]: {spline_kernel[i, :, 0]}")
#print(f"Anzahl der Punkte in spline_kernel[{i}, :, 0]: {len(spline_kernel[i, :, 0])}")

#1) Skalierung der Eingabe
scaled_input = freq_vector * scale_factor.T
print(scaled_input)

'''
[[ 0.0411432  0.037582   0.1758915  0.0461358 -0.0471355 -0.07038
  -0.2098314 -0.12574    0.045718   0.2131846  0.0936522  0.0842368
   0.606119  -0.109174  -0.1425015]]
'''
#2)Anwendung der Splines als Polynome
spline_output = np.zeros((15, 1))  #Output-Speicher für 15 Features

for i in range(15):
    poly = Polynomial(spline_kernel[i][:, 0])  #shape (15, 8, 1) -> Umwandlung auf (8,)
    spline_output[i] = poly(scaled_input[0, i])
'''
for i in range(15):
    poly = Polynomial(spline_kernel[i])  #Polynom 7. Grades erzeugen
    spline_output[i] = poly(scaled_input[0, i])  #Polynom auf den jeweiligen (also ersten auf ersten usw) skalierten Wert anwenden
'''
# 3) Summieren der Ergebnisse und Bias hinzufügen
output = np.sum(spline_output) + bias
print("Output von dem ersten polynom:", spline_output[0])
#Output von dem ersten polynom: [-2.691]
print("Model hidden layer Output:", output)
#Model hidden layer Output: [-3.1172]


######anwenden der output layer: 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#4) Multiplikation, Addition von bias und Sigmoid-Funktion anwenden
final_output = sigmoid(output * 11.785503 + 0.03887552)

print("Model Output:", final_output)
################

#2) Anwendung der Splines
#spline_output = np.zeros((15, 1))  #Output-Speicher für 15 Features

#i = int(1)
#for i in range(15):
#spline = CubicSpline(spline_grid[:12], spline_kernel[1, :12, 0])  # spline funktionen erzeugen
    #spline = CubicSpline(spline_grid, spline_kernel[i, :, 0])
    #spline = CubicSpline(spline_grid[:12], spline_kernel[i, :, 0])  # spline funktionen erzeugen
#spline_output[1] = spline(scaled_input[0, 1])  #Spline auf den skalierten Wert anwenden

# 3) Summieren der Ergebnisse und Bias hinzufügen
#output = np.sum(spline_output) + bias
#output = spline_output[1]+bias

#print("Model Output:", output)

