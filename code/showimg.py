import scipy.io
import numpy as np
import argparse
import matplotlib.pyplot as plt

# Argument-Parser erstellen
parser = argparse.ArgumentParser(description="Ein Programm, das einen Ordnerpfad als Argument annimmt.")

# Argument für den data_folder hinzufügen
parser.add_argument('file', type=str, help="Pfad zur Datei")
parser.add_argument('output_file', type=str, help="Pfad zur Output-Datei")

# Argumente parsen
args = parser.parse_args()

# Lade die .mat-Datei
mat_data = scipy.io.loadmat(args.file)

# Zeige den Inhalt der Datei an, um herauszufinden, welche Variablen sie enthält
print(mat_data.keys())

# # Wähle die relevante Variable, die das Bild enthält
# # Angenommen, die Variable heißt 'bild'
real_array = mat_data['i_real']
fake_array = mat_data['i_fake']

# # Überprüfe die Dimensionen des Bildes
print(fake_array.shape)

fake_array = np.squeeze(fake_array)  # Entferne die Batch-Dimension (Shape: (3, 196, 196))
fake_array = np.transpose(fake_array, (1, 2, 0))  # Ändere die Achsenreihenfolge zu (196, 196, 3)
real_array = np.squeeze(real_array)  # Entferne die Batch-Dimension (Shape: (3, 196, 196))
real_array = np.transpose(real_array, (1, 2, 0))  # Ändere die Achsenreihenfolge zu (196, 196, 3)

# # Zeige das Bild an
# plt.imsave(args.output_file, fake_array)
plt.imsave(args.output_file, real_array)
# plt.imshow(real_array, cmap='gray')  # Verwende 'cmap' für Graustufenbilder
# plt.imshow(fake_array, cmap='gray')  # Verwende 'cmap' für Graustufenbilder
# plt.colorbar()  # Zeige die Farbskala (falls gewünscht)
# plt.show()
