import my_labeling
import my_preprocessing
import glob


################################################################################
################################################################################

# This script is used to label and prepare all data coming from CarMaker
# for this purpose, this script uses methods from "my_labeling" and
# "my_preprocessing"

################################################################################
################################################################################


MIN_CONSECUTIVE_SCENES = 15    # minimum number of required consecutive scenes to be a scenario
SCENARIOS = ["free_cruising", "approaching", "following",
             "catching_up", "overtaking", "lane_change_left", "lane_change_right", "unknown"]

INPUT_DIR = "input"     # directory where CarMaker data is stored
OUTPUT_DIR = "output"   # output directory where labeled data should be stored


print("------------------ Prepare data... ------------------")
data_list = glob.glob(INPUT_DIR + "/data/*")
data_list.sort()
frames_list = glob.glob(INPUT_DIR + "/frames/*")
frames_list.sort()
assert data_list.__len__() == frames_list.__len__()
for i in range(data_list.__len__()):
    print("--- " + str(i+1) + "/" + str(data_list.__len__()) + " - " + data_list[i])
    data, metadata, all_vehicles, images = my_labeling.get_data(data_list[i], frames_list[i])
    scenarios_labels = my_labeling.label_scenarios(data, metadata, all_vehicles, images,
                                                   SCENARIOS, MIN_CONSECUTIVE_SCENES)
    my_preprocessing.prepare_images(scenarios_labels, images, SCENARIOS, MIN_CONSECUTIVE_SCENES, OUTPUT_DIR)
print("-----------------------------------------------------")
