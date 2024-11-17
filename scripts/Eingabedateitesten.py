import pandas as pd

file_path = '../data/processed/zone/train/train_zone_permuted_shuffled.csv'

#erwartete Anzahl Spalten
expected_columns = 16

#CSV-File chunkweise laden (laedt sonst ewig)
chunk_size = 10000 

#initialisieren der Variablen für Überprüfung
total_rows = 0
missing_values = None
unexpected_columns = False

try:
    chunk_iter = pd.read_csv(file_path, chunksize=chunk_size)
    for chunk in chunk_iter:
        #Gesamtzahl der geladenen Zeilen erhöhen
        total_rows += len(chunk)
        
        #Anzahl der Spalten im ersten Chunk
        if total_rows == len(chunk) and len(chunk.columns) != expected_columns:
            print(f"Unexpected number of columns: {len(chunk.columns)} (expected {expected_columns})")
            unexpected_columns = True
        
        #ersten Zeilen des ersten Chunks anzeigen
        if total_rows == len(chunk):
            print("First few rows of the data:\n", chunk.head())
        
        #fehlende Werte?
        if missing_values is None:
            missing_values = chunk.isnull().sum()
        else:
            missing_values += chunk.isnull().sum()

except Exception as e:
    print(f"Error while reading the file: {e}")

#Ergebnis:
if not unexpected_columns:
    print(f"Total rows loaded: {total_rows}")
    print("Missing values in each column:\n", missing_values)

