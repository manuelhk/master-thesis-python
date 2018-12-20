# Master Thesis: Künstliches Lernen: Trainingsdaten aus dem virtuellen Fahrversuch für Szenarienklassifizierung mit Deep Learning Algorithmen

## EN: “Artificial Learning“ - Training Data From a Virtual Driving Test for Scenario Classification With Deep Learning Algorithms

Dieser Code ist Teil meiner Masterarbeit am Karlsruher Institut für Technologie (KIT) am Institut für Technik der 
Informationsverarbeitung (ITIV).

## Abstract der Masterarbeit

Ab Stufe 3 des autonomen Fahrens kontrolliert nicht mehr der Fahrer, sondern das System die Umgebung in der sich ein Fahrzeug bewegt. Das Resultat ist, dass die Sicherheit von hochautomatisierten Fahrerassistenzsystemen (FAS) in Zukunft in allen potentiellen Szenarien garantiert sein muss. Da viele Szenarien bisher unbekannt sind, stellt diese Absicherung die Automobilindustrie vor große Herausforderungen. In dieser Arbeit wird ein Konzept für die Klassifizierung von Fahrszenarien entwickelt und umgesetzt. Dafür wird ein künstliches neuronales Netz mit 95% synthetischen und 5% realen Videodaten aus fünf Szenarienklassen trainiert. Diese synthetischen Daten werden zuvor mit der Simulationssoftware CarMaker generiert und automatisch gelabelt. Mit diesem Ansatz soll es in Zukunft möglich sein, einen Klassifikator mit allen bisher bekannten Szenarien zu trainieren. Darauf basierend kann dieser Klassifikator bekannte Szenarien erkennen und bisher unbekannte Szenarien identifizieren. Mit der Entwicklung des Ansatzes und einem Proof-of-Concept liefert diese Arbeit einen theoretischen und praktischen Beitrag zur Absicherung von hochautomatisierten FAS.

## Übersicht des Codes

Der Code kann grundsätzlich in zwei Teile eingeteilt werden. Im ersten Teil werden Daten, die mit der Simulationssoftware 
CarMaker generiert werden, gelabelt und für das Training mit neuronalen Netzen vorbereitet. Im zweiten Teil werden
neuronale Netze designed und mit den vorbereiteten Daten trainiert. Daneben gibt es ein Skript (`helper.py) um 
Trainingsergebnisse zu visualisieren.

Die Funktionsweise der beiden Teile wird in den folgenden Abschnitten erläutert.

## Benötigte Packages und Versionen

numpy 1.15.0

opencv 3.4.1

matplotlib 3.0.1

keras 2.2.2

## Teil 1: Vorbereitung der Trainingsdaten

Für die Vorbereitung der Trainingsdaten wird das Skript `main_preparation.py` mit Methoden aus `my_labeling.py` und 
`my_preprocessing.py` verwendet. Die Funktionsweise des Skripts wird in den folgenden Schritten erläutert. Der 
vollständige Code kann im Skript `main_preparation.py` mit Kommentaren gefunden werden.

### Schritt 1.1: Verzeichnisse vorbereiten

```python
import glob

INPUT_DIR = "input"
OUTPUT_DIR = "output"

data_list = glob.glob(INPUT_DIR + "/data/*")
data_list.sort()
frames_list = glob.glob(INPUT_DIR + "/frames/*")
frames_list.sort()
```

Im Input-Verzeichnis `INPUT_DIR` befinden sich die zwei Ordner "data" und "frames". Im "data"-Ordner sind 
alle .dat-Dateien aus CarMaker von jedem einzelnen TestRun mit aufsteigender Nummerierung gespeichert. Im "frames"-Ordner 
sind die dazugehörigen Bilder von jedem TestRun in einem separaten Ordner gespeichert. Diese Ordner sind mit der 
gleichen aufsteigenden Nummerierung versehen, um während des Labelns die .dat-Dateien den richtigen Bildern 
zuordnen zu können. Die Bilder in den jeweiligen Ordnern sind ebenfalls aufsteigend nummeriert und im .jpg-Format 
abgespeichert.

### Schritt 1.2: Daten eines TestRuns in den Arbeitsspeicher laden

```python
import my_labeling

data = data_list[0]
frames = frames_list[0]

data, metadata, all_vehicles, images = my_labeling.get_data(data, frames)
```

mit folgenden Variablen:

`String: data` - Pfad zur .dat-Ergebnisdatei die mit CarMaker generiert wurde

`String: frames` - Pfad zu dem Ordner in dem alle Bilder des dazugehörigen TestRuns mit aufsteigender Numerierung gespeichert sind

`Numpy array: data` - .dat-Ergebnisdatei ohne Header

`List: metadata` - Liste aller Variablen die mit CarMaker generiert wurden, in derselben Reihenfolge wie die Daten in "data"

`List: all_vehicles` - Liste aller Fahrzeugnamen (T0, T1, ...) die in dem TestRun erzeugt wurden

`List: images` - Liste von Pfaden (Strings) zu jedem Bild des TestRuns, aufsteigend geordnet

### Schritt 1.3: Szenarios labeln

```python
MIN_CONSECUTIVE_SCENES = 15
SCENARIOS = ["free_cruising", "approaching", "following", "catching_up", "overtaking", "lane_change_left", "lane_change_right", "unknown"]

scenarios_labels = my_labeling.label_scenarios(data, metadata, all_vehicles, images, SCENARIOS, MIN_CONSECUTIVE_SCENES)
```

`data, metadata, all_vehicles, images` - siehe oben

`List: SCENARIOS - Liste` (Strings) mit allen Szenarien

`Integer: MIN_CONSECUTIVE_SCENES` - Mindestanzahl von konsekutiven Szenen für ein Szenario (in meiner Arbeit immer 15)

`Numpy array: scenarios_labels` - Array der Dimension `(x, len(SCENARIOS))`. Dabei beschreibt x die Anzahl der Szenen in 
dem jeweiligen TestRun und `len(SCENARIOS)` die Anzahl der unterschiedlichen Szenarienklassen in der Liste `SCENARIOS`. In 
dem Array `scenarios_labels` wird für jede Szene x markiert (0: False, 1: True) zu welchen Szenarien diese Szene 
zugeordnet ist. Beispielsweise ist die Szene `(1, 0, 0, 0, 1, 0, 0, 0)` den Szenarien `SCENARIOS[0]` und 
`SCENARIOS[4]` zugeordnet. Eine "1" an der letzten Position bedeutet, dass die Szene als unbekannt gelabelt wurde.

### Schritt 1.4: Szenarios als numpy arrays speichern

```python
import my_preprocessing

my_preprocessing.prepare_images(scenarios_labels, images, SCENARIOS, MIN_CONSECUTIVE_SCENES, OUTPUT_DIR)
```

`scenarios_labels, images, SCENARIOS, MIN_CONSECUTIVE_SCENES` - siehe oben

`String: OUTPUT_DIR` - Im Output-Verzeichnis sind Unterordner für jede mögliche Szenarioklasse (aus `SCENARIOS`) vorhanden
. Die Methode `my_preprocessing.prepare_images(...)` speichert jedes Szenario als numpy array im jeweiligen Ordner der 
Klasse.


## Teil 2: Neuronale Netze trainieren

Für das Training der neuronalen Netze wird das Skript `main_training.py` mit Methoden aus `my_generator` und 
`my_model` verwendet. Die Funktionsweise des Skripts wird in den folgenden Schritten erläutert. Der vollständige Code 
kann im Skript `main_training.py` mit Kommentaren gefunden werden.

### Schritt 2.1: Verzeichnisse vorbereiten

```python
input_directory_sim = "input"
input_directory_real = "input/real"
output_directory = "output"
```

`String: input_directory_sim` - In diesem Verzeichnis sind die synthetischen Inputdaten in Unterordnern für jede 
Szenarioklasse abgespeichert. Damit hat dieses Verzeichnis die gleiche Struktur wie das Output-Verzeichnis von 
Schritt 1.4.

`String: input_directory_real` - Dieses Verzeichnis ist gleich strukturiert wie das vorherige Verzeichnis und die 
realen Input-Daten sind hier abgespeichert.

`String: output_directory` - In diesem Verzeichnis wird das trainierte Modell, die Konfiguration und die Ergebnisse
 der Testdaten gespeichert.

### Schritt 2.2: Neuronales Netz erstellen und kompilieren

```python
import my_model
import keras

classification = "video"
cnn_name = "inception_v3"
dropout = True

if classification == "video":
    model = my_model.build_video_model(SCENARIOS.__len__(), cnn_name, dropout)
if classification == "image":
    model = my_model.build_image_model(SCENARIOS.__len__(), cnn_name, dropout)

model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(1e-4), metrics=["accuracy"])
```

Das neuronale Netz kann unterschiedlich konfiguriert werden. Die Klassifizierung kann auf Basis von einzelnen Bildern 
`classification = "image"` oder auf Basis von Bildsequenzen `classification = "video"` erfolgen. Weiterhin kann zwischen 
zwei unterschiedlichen CNNs (Inception-V3 und Xception) für die Extraktion von räumlichen Merkmalen unterschieden werden. 
Zusätzlich kann die vorletzte Schicht mit oder ohne Dropout konfiguriert werden.

### Schritt 2.3: Datensätze für Training, Validierung und Test vorbereiten

```python
import my_generator

max_sim_data_per_class = 950
max_real_data_per_class = 67 

SCENARIOS = ["free_cruising", "following", "catching_up", "lane_change_left", "lane_change_right"]
PARAMS = {'dim': dim,
          'batch_size': 4,
          'n_classes': SCENARIOS.__len__(),
          'n_channels': 3,
          'shuffle': True,
          'cnn_name': cnn_name,
          'classification': classification}

train_sim, val_sim, test_sim, label_dict = my_generator.get_data_and_labels(input_directory_sim, SCENARIOS,
                                                                            max_number=max_sim_data_per_class,
                                                                            train_share=0.70, val_share=0.90)
train_real, val_real, test_real, label_real = my_generator.get_data_and_labels(input_directory_real, SCENARIOS,
                                                                               max_number=max_real_data_per_class,
                                                                               train_share=0.50, val_share=0.75)

train_list = train_sim + train_real
val_list = val_sim + val_real
label_dict.update(label_real)

random.shuffle(train_list)
random.shuffle(val_list)

train_generator = my_generator.DataGenerator(train_list, label_dict, **PARAMS)
val_generator = my_generator.DataGenerator(val_list, label_dict, **PARAMS)
```

In diesem Schritt werden mit der Methode `my_generator.get_data_and_labels(...)`, jeweils für die synthetischen und 
realen Daten, eine Liste für das Training, die Validierung und den Test erstellt. Dann werden die Listen zusammengefasst 
und jeweils für das Training und die Validierung während dem Training ein DataGenerator erstellt. Dieser DataGenerator 
erstellt einen Datenstrom (engl. data stream) zum neuronalen Netz während des Trainings. Dies ist notwendig, weil nicht 
alle Daten für das Training in den Arbeitsspeicher geladen werden können. 

### Schritt 2.4: Training des neuronalen Netzes

```python
import numpy as np

history = model.fit_generator(generator=train_generator, validation_data=val_generator, epochs=epochs)

model.save(output_directory + "/model.h5")
np.save(output_directory + "/history.npy", history)
```

In diesem Schritt wird das neuronale Netz mit den oben generierten Daten trainiert und validiert. Zurückgegeben wird ein 
`history` Objekt, das u.a. den Fehler und die Genauigkeit nach jeder Epoche speichert. Das trainierte Modell und das 
`history` Objekt werden im Output-Verzeichnis gespeichert.

### Schritt 2.5: Test des trainierten Netzes

```python
pred_real = []
for path in test_real:
    pred_real.append(my_generator.get_prediction(model, path, cnn_name, classification))
pred_real = np.squeeze(np.array(pred_real))
np.save(output_directory + "/predictions_test_data_real.npy", pred_real)
```

In diesem Schritt werden die realen Testdaten (analog dazu auch die synthetischen Testdaten) von dem trainierten 
Modell klassifiziert und die Ergebnisse werden im Output-Verzeichnis gespeichert.


### Schritt 2.6: Visualisierung mit dem Skript `helper.py`

Die Methoden im Skript `helper.py` können für die Visualisierung der Genauigkeit (engl. accuracy), des Fehlers 
während dem Training (engl. loss) und der Konfusionsmatrix (engl. confusion matrix) verwendet werden. Für die Verwendung 
der Methoden müssen ggf. die Pfade in den Methoden angepasst werden.
