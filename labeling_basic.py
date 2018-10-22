import numpy as np
import glob
import my_vehicle


DATA_HZ = 50
FPS = 5
SMOOTHING_FACTOR = 150   # minimum number of required consecutive scenes to be a scenario
SCENARIOS = {"FREE_CRUISING": 0, "APPROACHING": 1, "FOLLOWING": 2,
             "CATCHING_UP": 3, "OVERTAKING": 4, "LANE_CHANGE_LEFT": 5, "LANE_CHANGE_RIGHT": 6,
             "V2_CATCHING_UP": 7, "V2_OVERTAKING": 8, "UNKNOWN": 9}


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
        data_index = int(index * DATA_HZ / 5)
        labels = np.zeros(SCENARIOS.__len__())
        ego_vehicle = get_ego_vehicle(data, metadata, data_index)
        relevant_vehicles = get_relevant_vehicles(data[data_index, :], metadata, all_vehicles, ego_vehicle)

        labels[0] = free_cruising_fn(data[data_index, :], metadata, relevant_vehicles)
        labels[1] = approaching_fn(data, metadata, relevant_vehicles, ego_vehicle, data_index)
        labels[2] = following_fn(relevant_vehicles, ego_vehicle)
        labels[3] = catching_up_fn(relevant_vehicles, ego_vehicle)
        labels[4] = overtaking_fn(relevant_vehicles, ego_vehicle)
        labels[5] = lane_change_left_fn(data, metadata, data_index)
        labels[6] = lane_change_right_fn(data, metadata, data_index)
        labels[7] = v2_catching_up_fn(relevant_vehicles, ego_vehicle)
        labels[8] = v2_overtaking_fn(relevant_vehicles, ego_vehicle)
        labels[9] = unknown_fn(labels)
        # scenarios_labeled = smoothing_fn(scenes_labeled)
        # scenarios_labeled[:, 9] = unknown_fn(scenarios_labeled)
        label_dict.update({image_path: labels})
        index += 1
    return label_dict


def get_ego_vehicle(data, metadata, index):
    ego_lane_id = data[index, metadata.index("Car.Road.Lane.Act.LaneId")]
    ego_s_road = data[index, metadata.index("Car.Road.sRoad")]
    ego_v = data[index, metadata.index("Car.v")]
    s_0 = data[index, metadata.index("Car.v")] * 3.6
    s_1 = data[index, metadata.index("Car.v")] * 3.6 * 2 / 3
    s_2 = data[index, metadata.index("Car.v")] * 3.6 * 1 / 2
    s_3 = data[index, metadata.index("Car.v")] * 3.6 * 1 / 3
    ego_vehicle = my_vehicle.Vehicle("ego", ego_lane_id, ego_s_road, v=ego_v, s_0=s_0, s_1=s_1, s_2=s_2, s_3=s_3)
    return ego_vehicle


def get_relevant_vehicles(data, metadata, all_vehicles, ego_vehicle):
    relevant_vehicles = []
    for vehicle in all_vehicles:
        ds = data[metadata.index("Sensor.Object.OB01.Obj." + vehicle + ".NearPnt.ds_p")]
        if ds != 0 and abs(ds) < ego_vehicle.s_0:
            lane_id = data[metadata.index("Traffic." + vehicle + ".Lane.Act.LaneId")]
            s_road = data[metadata.index("Traffic." + vehicle + ".sRoad")]
            dv = data[metadata.index("Sensor.Object.OB01.Obj." + vehicle + ".NearPnt.dv_p")]
            new_vehicle = my_vehicle.Vehicle(vehicle, lane_id, s_road, ds=ds, dv=dv)
            relevant_vehicles.append(new_vehicle)
    return relevant_vehicles


def free_cruising_fn(data, metadata, relevant_vehicles):
    if (relevant_vehicles.__len__() == 0 and
            data[metadata.index("Car.v")] > 17):
        return SCENARIOS["FREE_CRUISING"]
    return SCENARIOS["UNKNOWN"]


def approaching_fn(data, metadata, relevant_vehicles, ego_vehicle, index):
    lower_index = max(index - 50, 0)
    if lower_index == 0:
        ego_v_mean = data[index, metadata.index("Car.v")]
    else:
        ego_v_mean = data[lower_index:index, metadata.index("Car.v")].mean()
    for vehicle in relevant_vehicles:
        if (ego_vehicle.v < ego_v_mean and
                ego_vehicle.s_2 < vehicle.ds < ego_vehicle.s_0 and
                ego_vehicle.s_road < vehicle.s_road and
                ego_vehicle.lane_id == vehicle.lane_id):
            return SCENARIOS["APPROACHING"]
    return SCENARIOS["UNKNOWN"]


def following_fn(relevant_vehicles, ego_vehicle):
    for vehicle in relevant_vehicles:
        if (vehicle.dv < ego_vehicle.v * 0.05 and
                ego_vehicle.s_3 < vehicle.ds < ego_vehicle.s_1 and
                ego_vehicle.s_road < vehicle.s_road and
                ego_vehicle.lane_id == vehicle.lane_id):
            return SCENARIOS["FOLLOWING"]
    return SCENARIOS["UNKNOWN"]


def catching_up_fn(relevant_vehicles, ego_vehicle):
    for vehicle in relevant_vehicles:
        if (vehicle.dv < 0 and
                0 <= vehicle.ds < ego_vehicle.s_0 and
                ego_vehicle.s_road <= vehicle.s_road and
                ego_vehicle.lane_id == vehicle.lane_id - 1):
            return SCENARIOS["CATCHING_UP"]
    return SCENARIOS["UNKNOWN"]


def overtaking_fn(relevant_vehicles, ego_vehicle):
    for vehicle in relevant_vehicles:
        if (vehicle.dv > 0 and
                0 < vehicle.ds < ego_vehicle.s_0 and
                vehicle.s_road < ego_vehicle.s_road and
                ego_vehicle.lane_id == vehicle.lane_id - 1):
            return SCENARIOS["OVERTAKING"]
    return SCENARIOS["UNKNOWN"]


def lane_change_left_fn(data, metadata, index):
    lower_index = max(index - 200, 0)
    first_lane = data[lower_index, metadata.index("Car.Road.Lane.Act.LaneId")]
    for i in range(lower_index, index):
        second_lane = data[i, metadata.index("Car.Road.Lane.Act.LaneId")]
        if first_lane == second_lane + 1:
            return SCENARIOS["LANE_CHANGE_LEFT"]
    return SCENARIOS["UNKNOWN"]


def lane_change_right_fn(data, metadata, index):
    lower_index = max(index - 200, 0)
    first_lane = data[lower_index, metadata.index("Car.Road.Lane.Act.LaneId")]
    for i in range(lower_index, index):
        second_lane = data[i, metadata.index("Car.Road.Lane.Act.LaneId")]
        if first_lane == second_lane - 1:
            return SCENARIOS["LANE_CHANGE_RIGHT"]
    return SCENARIOS["UNKNOWN"]


def v2_catching_up_fn(relevant_vehicles, ego_vehicle):
    for vehicle in relevant_vehicles:
        s_0_v2 = (vehicle.dv + ego_vehicle.v) * 3.6
        if (0 < vehicle.dv and
                0 <= vehicle.ds < s_0_v2 and
                vehicle.s_road <= ego_vehicle.s_road and
                vehicle.lane_id == ego_vehicle.lane_id - 1):
            return SCENARIOS["V2_CATCHING_UP"]
    return SCENARIOS["UNKNOWN"]


def v2_overtaking_fn(relevant_vehicles, ego_vehicle):
    for vehicle in relevant_vehicles:
        s_0_v2 = (vehicle.dv + ego_vehicle.v) * 3.6
        if (0 < vehicle.dv and
                0 < vehicle.ds < s_0_v2 and
                ego_vehicle.s_road < vehicle.s_road and
                vehicle.lane_id == ego_vehicle.lane_id - 1):
            return SCENARIOS["V2_OVERTAKING"]
    return SCENARIOS["UNKNOWN"]


def unknown_fn(label):
    return 1
