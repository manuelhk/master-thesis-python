# Master Thesis: Künstliches Lernen: Trainingsdaten aus dem virtuellen Fahrversuch für Szenarienklassifizierung mit Deep Learning Algorithmen

## EN: “Artificial Learning“ - Training Data From a Virtual Driving Test for Scenario Classification With Deep Learning Algorithms

Dieser Code ist Teil meiner Masterarbeit am Karlsruher Institut für Technologie (KIT) am Institut für Technik der 
Informationsverarbeitung (ITIV).

## Abstract der Masterarbeit

Abstract...

## Übersicht des Codes

Der Code kann grundsätzlich in zwei Teile eingeteilt werden. In einem Teil werden Daten, die mit der Simulationssoftware 
CarMaker generiert werden, gelabelt und für das Training mit neuronalen Netzen vorbereitet. Im zweiten Teil werden
neuronale Netze designed und mit diesen Daten trainiert. Daneben gibt es ein Skript (helper.py) um Trainingsergebnisse
zu visualisieren.

Die Funktionsweise dieser beiden Teile und das Skript helper.py wird in den folgenden Abschnitten erläutert.

## Benötigte Packages und Versionen

numpy 1.15.0

opencv 3.4.1

matplotlib 3.0.1

keras 2.2.2

## Teil I: Vorbereitung der Trainingsdaten

Für die Vorbereitung der Trainingsdaten wird das Skript "main_preparation.py" mit Methoden von "my_labeling.py" und 
"my_preprocessing.py" verwendet. Um das Skript zu verwenden, können die folgende Schritte angewendet werden.

### Schritt 1: Verzeichnisse vorbereiten

```python
import glob

INPUT_DIR = "input"
OUTPUT_DIR = "output"

data_list = glob.glob(INPUT_DIR + "/data/*")
data_list.sort()
frames_list = glob.glob(INPUT_DIR + "/frames/*")
frames_list.sort()
```

Das Input-Verzeichnis (INPUT_DIR) und das Output-Verzeichning (OUTPUT_DIR) müssen
definiert werden. Im Input-Verzeichnis müssen die zwei Ordner "data" und "frames" existieren. Im "data"-Ordner müssen 
alle .dat-Dateien aus CarMaker von jedem einzelnen TestRun mit aufsteigender Nummerierung liegen. Im "frames"-Ordner 
müssen die dazugehörigen Bilder von jedem TestRun in einem separaten Ordner abgelegt werden. Diese Ordner müssen mit der 
gleichen aufsteigenden Nummerierung versehen werden, um während des Labelings die .dat-Dateien den richtigen Bildern 
zuordnen zu können. Die Bilder in den jeweiligen Ordnern müssen ebenfalls aufsteigend nummeriert und im .jpg-Format 
abgespeichert werden.

### Schritt 2: Daten eines TestRuns in den Arbeitsspeicher laden

```python
import my_labeling

data = data_list[0]
frames = frames_list[0]

data, metadata, all_vehicles, images = my_labeling.get_data(data, frames)
```

mit folgenden Variablen:

``String: data`` - Pfad zur .dat-Ergebnisdatei die mit CarMaker generiert wurde

``String: frames`` - Pfad zu dem Ordner in dem alle Bilder des dazugehörigen TestRuns mit aufsteigender Numerierung gespeichert sind

``Numpy array: data`` - .dat-Ergebnisdatei ohne Header

``List: metadata`` - Liste aller Variablennamen die mit CarMaker generiert wurden in der selben Reihenfolge wie die Daten in "data"

``List: all_vehicles`` - Liste aller Fahrzeugnamen (T0, T1, ...) die in dem TestRun erzeugt wurden

``List: images`` - Liste von Pfaden (Strings) zu jedem Bild des TestRuns, aufsteigend geordnet

### Schritt 3: Szenarios labeln

```python
MIN_CONSECUTIVE_SCENES = 15
SCENARIOS = ["free_cruising", "approaching", "following", "catching_up", "overtaking", "lane_change_left", "lane_change_right", "unknown"]

scenarios_labels = my_labeling.label_scenarios(data, metadata, all_vehicles, images, SCENARIOS, MIN_CONSECUTIVE_SCENES)
```

``data, metadata, all_vehicles, images`` - siehe oben

``List: SCENARIOS - Liste`` (Strings) mit allen Szenarien

``Integer: MIN_CONSECUTIVE_SCENES`` - Mindestanzahl von konsekutiven Szenen für ein Szenario (in meiner Arbeit immer 15)

`Numpy array: scenarios_labels` - Array der Dimension (x, len(SCENARIOS)). Dabei beschreibt x die Anzahl der Szenen in 
dem jeweiligen TestRun und len(SCENARIOS) die Anzahl der Szenarien in der Liste SCENARIOS. In dem Array scenarios_labels 
wird für jede Szene x markiert (0: False, 1: True) zu welchen Szenarien diese Szene zugeordnet werden kann. 
Beispielsweise wird die Szene (1, 0, 0, 0, 1, 0, 0, 0) den Szenarien SCENARIOS[0] und SCENARIOS[4] zugeordnet. Eine "1" 
an der letzten Position bedeutet, dass die Szene unbekannt ist.

