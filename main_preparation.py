import labeling_basic as lb
import preprocessing as pp
import glob


MIN_CONSECUTIVE_SCENES = 15    # minimum number of required consecutive scenes to be a scenario
SCENARIOS = ["free_cruising", "approaching", "following",
             "catching_up", "overtaking", "lane_change_left", "lane_change_right",
             "v2_catching_up", "v2_overtaking", "unknown"]

INPUT_DIR = "input"
OUTPUT_DIR = "output"


def prepraration():
    print("------------------ Prepare data... ------------------")
    data_list = glob.glob(INPUT_DIR + "/data/*")
    data_list.sort()
    frames_list = glob.glob(INPUT_DIR + "/frames/*")
    frames_list.sort()
    assert data_list.__len__() == frames_list.__len__()
    for i in range(data_list.__len__()):
        print("--- " + str(i+1) + "/" + str(data_list.__len__()) + " - " + data_list[i])
        data, metadata, all_vehicles, images = lb.get_data(data_list[i], frames_list[i])
        scenarios_labels = lb.label_scenarios(data, metadata, all_vehicles, images, SCENARIOS, MIN_CONSECUTIVE_SCENES)
        # lb.save_video(label_dict, VIDEO_PATH, SCENARIOS)
        pp.prepare_images(scenarios_labels, images, SCENARIOS, MIN_CONSECUTIVE_SCENES, OUTPUT_DIR)
    print("-----------------------------------------------------")
    pass


prepraration()
