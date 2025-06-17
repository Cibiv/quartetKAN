import numpy as np
import pandas as pd
import tensorflow.keras as keras
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Model to test.")
parser.add_argument("-t", "--test", default='../data/processed/zone/test/1000bp', help="Path to test directory")
parser.add_argument("-o", "--offset", type=int, default=4, help="Index of first feature column.")
parser.add_argument("-n", "--no_feat", type=int, default=15, help="Number of feature columns.")
args = vars(parser.parse_args())

#Modell laden
model = keras.saving.load_model(args['model'])
model.summary()

#Testdatei auswählen
test_files = [f for f in os.listdir(args["test"]) if f.endswith('.csv')]
if not test_files:
    raise FileNotFoundError("Keine Testdateien gefunden!")

test_file = test_files[0]  # Erste Datei aus dem Verzeichnis nehmen
df = pd.read_csv(f'{args["test"]}/{test_file}')

#beispiel-Frequenzvektor extrahieren (erste Zeile der Feature Spalten)
offset = args["offset"]
no_feat = args["no_feat"]
feature_vector = df.iloc[0, offset:offset + no_feat].values  #erster vektor

#in numpy-Array umwandeln und die Form anpassen
feature_vector = np.array(feature_vector, dtype=np.float32).reshape(1, -1)

#modell anwenden
output = model.predict(feature_vector)

#ergebnis 
print("Beispiel-Frequenzvektor:", feature_vector)
print("Model Output:", output)

#anschauen was es für layer gibt: 
#model.summary()
#auf den i-ten layer des models mit models.layers[i] zugreifen
#hidden layer:
#print(model.layers[0])
#<keras.src.saving.legacy.saved_model.load.DenseKAN object at 0x7f4d8748d850>

#output layer:
#print(model.layers[1])
#<keras.src.layers.core.dense.Dense object at 0x7fdf76b06390>

#für dense layers erhält man die weights mit model.layers[i].weights
#print(model.layers[1].weights)
