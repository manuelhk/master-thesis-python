import labeling_basic as lb

data, metadata, all_vehicles, images = lb.get_data("data/data/0808/data.dat", "data/data/0808/frames")
label_dict, scenarios_dict = lb.label_scenarios(data, metadata, all_vehicles, images)
lb.save_video(label_dict, "data/data/0808/video_new.avi")
