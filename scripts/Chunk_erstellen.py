import pandas as pd


def create_single_chunk(input_file, output_file, num_lines=1000):
    try:
        #laden der ersten eintausend Zeilen der Datei
        chunk = pd.read_csv(input_file, nrows=num_lines)
        #speichern des Chunks in die Ausgabedatei
        chunk.to_csv(output_file, index=False)
        print(f"Chunk mit {num_lines} Zeilen erfolgreich gespeichert unter: {output_file}")
    except Exception as e:
        print(f"Fehler beim Erstellen des Chunks: {e}")

# Beispiel: Aufruf der Funktion
file_path = '../data/processed/zone/train/train_zone_permuted_shuffled.csv'
output_path = '../data/processed/zone/train/chunk_1000.csv'
create_single_chunk(file_path, output_path)


