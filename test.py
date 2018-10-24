import labeling_basic as lb
import preprocessing as pp


MIN_CONSECUTIVE_SCENES = 15    # minimum number of required consecutive scenes to be a scenario
SCENARIOS = ["free_cruising", "approaching", "following",
             "catching_up", "overtaking", "lane_change_left", "lane_change_right",
             "v2_catching_up", "v2_overtaking", "unknown"]

ROOT_PATH_DATA = "data"
ROOT_PATH_FRAMES = "data"
FRAMES_PATH = "data/frames"
VIDEO_PATH = "data/video2.avi"
OUT_PATH = "test"


""" Labeling data..."""
data, metadata, all_vehicles, images = lb.get_data(DATA_PATH, FRAMES_PATH)
label_dict, label_np = lb.label_scenarios(data, metadata, all_vehicles, images, SCENARIOS, MIN_CONSECUTIVE_SCENES)
# lb.save_video(label_dict, VIDEO_PATH, SCENARIOS)
pp.prepare_images(label_np, images, SCENARIOS, MIN_CONSECUTIVE_SCENES, OUT_PATH)
