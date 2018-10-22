import labeling_basic as lb

data, metadata, all_vehicles, images = lb.get_data("data/data/0808/data.dat", "data/data/0808/frames")
label_dict, scenes_labels, scenarios_labels = lb.label_scenarios(data, metadata, all_vehicles, images)
