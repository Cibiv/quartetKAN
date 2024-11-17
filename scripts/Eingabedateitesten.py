import pandas as pd

#lade das CSV-File
df = pd.read_csv('../data/processed/zone/train/train_zone_permuted_shuffled.csv')

#prüfe die Anzahl der Spalten
expected_columns = 16
if len(df.columns) != expected_columns:
    print(f"Unexpected number of columns: {len(df.columns)} (expected {expected_columns})")

#prüfe die ersten Zeilen des DataFrames
print(df.head())
#prüfe auf fehlende Werte
missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)
