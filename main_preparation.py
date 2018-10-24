import labeling_basic as lb


MIN_CONSECUTIVE_SCENES = 15    # minimum number of required consecutive scenes to be a scenario
SCENARIOS = ["FREE_CRUISING", "APPROACHING", "FOLLOWING",
             "CATCHING_UP", "OVERTAKING", "LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT",
             "V2_CATCHING_UP", "V2_OVERTAKING", "UNKNOWN"]


""" Labeling data..."""
data, metadata, all_vehicles, images = lb.get_data("data/data/0808/data.dat", "data/data/0808/frames")
label_dict = lb.label_scenarios(data, metadata, all_vehicles, images, SCENARIOS, MIN_CONSECUTIVE_SCENES)
lb.save_video(label_dict, "data/data/0808/video_new.avi", SCENARIOS)
