import numpy as np
import glob
import my_vehicle



DATA_HZ = 50
FPS = 5
SMOOTHING_FACTOR = 150   # minimum number of required consecutive scenes to be a scenario
SCENARIOS = ["FREE_CRUISING", "APPROACHING", "FOLLOWING",
             "CATCHING_UP", "OVERTAKING", "LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT",
             "V2_CATCHING_UP", "V2_OVERTAKING", "UNKNOWN"]


def get_data(data_path, frames_path):
    """ Import data from path """
    data = np.genfromtxt(data_path)
    d = np.genfromtxt(data_path, comments=None, dtype=str, max_rows=1)
    d = d[1:]
    metadata = []
    for element in d:
        metadata.append(element)
    all_vehicles = []
    for text in metadata:
        if text.startswith("Traffic.") and text.endswith(".LaneId"):
            all_vehicles.append(text.split(".").__getitem__(1))
    images = glob.glob(frames_path + "/*.jpg")
    images.sort()
    return data, metadata, all_vehicles, images


def label_scenarios(data, metadata, all_vehicles, images):
    label_dict = dict()
    index = 0
    for image_path in images:
        label = np.zeros(SCENARIOS.__len__())
        data_index = int(index*DATA_HZ/5)
        label[0] = free_cruising_fn(data[data_index, :], metadata)
        label[1] = approaching_fn(data[data_index, :], metadata)
        label[2] = following_fn(data[data_index, :], metadata)
        label[3] = catching_up_fn(data[data_index, :], metadata)
        label[4] = overtaking_fn(data[data_index, :], metadata)
        label[5] = lane_change_left_fn(data[data_index, :], metadata)
        label[6] = lane_change_right_fn(data[data_index, :], metadata)
        label[7] = v2_catching_up_fn(data[data_index, :], metadata)
        label[8] = v2_overtaking_fn(data[data_index, :], metadata)
        label[9] = data_index
        #label[9] = unknown_fn(label)
        # scenarios_labeled = smoothing_fn(scenes_labeled)
        # scenarios_labeled[:, 9] = unknown_fn(scenarios_labeled)
        label_dict.update({image_path: label})
        index += 1
    return label_dict


def get_ego_vehicle_data(data, metadata, index):
    s_0 = data[index, metadata.index("Car.v")] * 3.6
    s_1 = data[index, metadata.index("Car.v")] * 3.6 * 2 / 3
    s_2 = data[index, metadata.index("Car.v")] * 3.6 * 1 / 2
    s_3 = data[index, metadata.index("Car.v")] * 3.6 * 1 / 3
    ego_v = data[index, metadata.index("Car.v")]
    return s_0, s_1, s_2, s_3, ego_v


def get_relevant_vehicles(data, metadata, all_vehicles, s_1, index):
    relevant_vehicles = []
    for vehicle in all_vehicles:
        ds = data[index, metadata.index("Sensor.Object.OB01.Obj." + vehicle + ".NearPnt.ds_p")]
        if ds != 0 and abs(ds) < s_1:
            dv = data[index, metadata.index("Sensor.Object.OB01.Obj." + vehicle + ".NearPnt.dv_p")]
            lane_id = data[index, metadata.index("Traffic." + vehicle + ".Lane.Act.LaneId")]
            s_road = data[index, metadata.index("Traffic." + vehicle + ".sRoad")]
            new_vehicle = my_vehicle.Vehicle(vehicle, ds, dv, lane_id, s_road)
            relevant_vehicles.append(new_vehicle)
    return relevant_vehicles


def free_cruising_fn(data, metadata):
    return 0


def approaching_fn(data, metadata):
    return 0


def following_fn(data, metadata):
    return 0


def catching_up_fn(data, metadata):
    return 0


def overtaking_fn(data, metadata):
    return 0


def lane_change_left_fn(data, metadata):
    return 0


def lane_change_right_fn(data, metadata):
    return 0


def v2_catching_up_fn(data, metadata):
    return 0


def v2_overtaking_fn(data, metadata):
    return 0


def unknown_fn(data, metadata):
    return 1
