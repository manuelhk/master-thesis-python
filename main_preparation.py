import labeling_basic as lb


MIN_CONSECUTIVE_SCENES = 15    # minimum number of required consecutive scenes to be a scenario
SCENARIOS = ["FREE_CRUISING", "APPROACHING", "FOLLOWING",
             "CATCHING_UP", "OVERTAKING", "LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT",
             "V2_CATCHING_UP", "V2_OVERTAKING", "UNKNOWN"]
DATA_PATH = "data/Highway_2/00_Highway_2_data.dat"
FRAMES_PATH = "data/Highway_2"
VIDEO_PATH = "data/Highway_1/video.avi"


""" Labeling data..."""
data, metadata, all_vehicles, images = lb.get_data(DATA_PATH, FRAMES_PATH)
# label_dict = lb.label_scenarios(data, metadata, all_vehicles, images, SCENARIOS, MIN_CONSECUTIVE_SCENES)
# lb.save_video(label_dict, VIDEO_PATH, SCENARIOS)
