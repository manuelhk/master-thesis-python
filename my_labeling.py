import numpy as np
import glob
import my_vehicle


################################################################################
################################################################################

# This is the main labeling script containing all necessary methods to label scenes
# (frames) from CarMaker with corresponding signal data

# The general idea is to create a vehicle object for the ego vehicle and all
# other vehicles for every scene, containing all necessary information from
# CarMaker such as velocity or distance. Then each scene is labeled by comparing
# specific values from each vehicle object

################################################################################
################################################################################


def get_data(data_path, frames_path):
    """
    This methods imports all data from one CarMaker TestRun

    :param data_path: path to the *.dat-file generated by a CarMaker TestRun
    :param frames_path: path to a folder containing all frames from that TestRun
    :return:
        - data array: all signal data from that TestRun
        - metdata array: names of each signal
        - all_vehicles list: names of each vehicle (e.g. T0) except the ego vehicle
        - images list: paths to all images from that TestRun
    """
    # print("Import data: " + data_path)
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


def label_scenarios(data, metadata, all_vehicles, images, scenarios, min_consecutive_scenes):
    """
    Based on specific rules, this methods labels all scenes within one TestRun of CarMaker
    Therefore, the objects produced in the method above are used

    :param data: array with all signal data from that TestRun
    :param metadata: array with names of each signal
    :param all_vehicles: list with names of each vehicle (e.g. T0) except the ego vehicle
    :param images: list with paths to all images from that TestRun
    :param scenarios: list with all names of the scenarios that have to be labeled
    :param min_consecutive_scenes: integer that specifies the number of consecutive scenes to be a scenario
    :return: array with the shape (x, number_of_scenarios) in which every row describes a scene and marks (0 or 1) if
             that scene is of one type or not
    """
    rows, columns = data.shape
    # print("Label data...")
    scenes_labels = np.zeros((images.__len__(), scenarios.__len__()))
    for i, image_path in enumerate(images):
        if i >= rows:
            break
        # if i % 100 == 0:
        #    print("Scenes: " + str(i) + "/" + str(images.__len__()))
        ego_vehicle = get_ego_vehicle(data, metadata, i)
        relevant_vehicles = get_relevant_vehicles(data[i, :], metadata, all_vehicles, ego_vehicle)

        scenes_labels[i, 0] = free_cruising_fn(data[i, :], metadata, relevant_vehicles)
        scenes_labels[i, 1] = approaching_fn(data, metadata, relevant_vehicles, ego_vehicle, i)
        scenes_labels[i, 2] = following_fn(relevant_vehicles, ego_vehicle)
        scenes_labels[i, 3] = catching_up_fn(relevant_vehicles, ego_vehicle)
        scenes_labels[i, 4] = overtaking_fn(relevant_vehicles, ego_vehicle)
        scenes_labels[i, 5] = lane_change_left_fn(data, metadata, i)
        scenes_labels[i, 6] = lane_change_right_fn(data, metadata, i)
        scenes_labels[i, 9] = unknown_fn(scenes_labels[i, :])

    scenarios_labels = smoothing_fn(scenes_labels, min_consecutive_scenes)
    label_dict = dict()
    for j, image_path in enumerate(images):
        scenarios_labels[j, 9] = unknown_fn(scenarios_labels[j, :8])
        label_dict.update({image_path: scenarios_labels[j, :]})
    return scenarios_labels


def get_ego_vehicle(data, metadata, index):
    """
    This method extracts all values from the data array that describe the ego vehicle
    with these values, a vehicle object is created

    :param data: array with all signal data from that TestRun
    :param metadata: array with names of each signal
    :param index: row index within the data array (index of one specific scene)
    :return: vehicle object containing all necessary values from the ego vehicle in this scene (index)
    """
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
    """
    This method creates a list of all vehicles that are close (0 - s_0)[m] to the ego vehcile.
    Therefore all signals belonging to close vehicles are extracted from the data array, and vehicle objects are
    created respectively

    :param data: array with all signal data from that TestRun
    :param metadata: array with names of each signal
    :param all_vehicles: list with names of each vehicle (e.g. T0) except the ego vehicle
    :param ego_vehicle: vehicle object containing all necessary values from the ego vehicle in this scene (index)
    :return: list of vehicle objects containing all vehicles that are close to the ego vehicle
    """
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
    """
    This method determines if the current scene is "free_cruising"

    :param data: array with all signal data from that TestRun
    :param metadata: array with names of each signal
    :param relevant_vehicles: list of vehicle objects containing all vehicles that are close to the ego vehicle
    :return: integer: 1 if the current scene is "free_cruising", 0 if it's not
    """
    if (relevant_vehicles.__len__() == 0 and
            data[metadata.index("Car.v")] > 17):
        return 1
    return 0


def approaching_fn(data, metadata, relevant_vehicles, ego_vehicle, index):
    """
    This method determines if the current scene (index) is "approaching"

    :param data: array with all signal data from that TestRun
    :param metadata: array with names of each signal
    :param relevant_vehicles: list of vehicle objects containing all vehicles that are close to the ego vehicle
    :param ego_vehicle: vehicle object containing all necessary values from the ego vehicle in this scene (index)
    :param index: current row in the data array, indicating the current scene
    :return: integer: 1 if the current scene is "approaching", 0 if it's not
    """
    lower_index = max(index - 15, 0)
    if lower_index == 0:
        ego_v_mean = data[index, metadata.index("Car.v")]
    else:
        ego_v_mean = data[lower_index:index, metadata.index("Car.v")].mean()
    for vehicle in relevant_vehicles:
        if (ego_vehicle.v < ego_v_mean and
                ego_vehicle.s_2 < vehicle.ds < ego_vehicle.s_0 and
                ego_vehicle.s_road < vehicle.s_road and
                ego_vehicle.lane_id == vehicle.lane_id):
            return 1
    return 0


def following_fn(relevant_vehicles, ego_vehicle):
    """
    This method determines if the current scene is "following"

    :param relevant_vehicles: list of vehicle objects containing all vehicles that are close to the ego vehicle
    :param ego_vehicle: vehicle object containing all necessary values from the ego vehicle in this scene (index)
    :return: integer: 1 if the current scene is "following", 0 if it's not
    """
    for vehicle in relevant_vehicles:
        if (vehicle.dv < ego_vehicle.v * 0.05 and
                ego_vehicle.s_3 < vehicle.ds < ego_vehicle.s_1 and
                ego_vehicle.s_road < vehicle.s_road and
                ego_vehicle.lane_id == vehicle.lane_id):
            return 1
    return 0


def catching_up_fn(relevant_vehicles, ego_vehicle):
    """
    This method determines if the current scene is "catching_up"

    :param relevant_vehicles: list of vehicle objects containing all vehicles that are close to the ego vehicle
    :param ego_vehicle: vehicle object containing all necessary values from the ego vehicle in this scene (index)
    :return: integer: 1 if the current scene is "catching_up", 0 if it's not
    """
    for vehicle in relevant_vehicles:
        if (vehicle.dv < 0 and
                0 <= vehicle.ds < ego_vehicle.s_0 and
                ego_vehicle.s_road <= vehicle.s_road and
                ego_vehicle.lane_id == vehicle.lane_id - 1):
            return 1
    return 0


def overtaking_fn(relevant_vehicles, ego_vehicle):
    """
    This method determines if the current scene is "overtaking"

    :param relevant_vehicles: list of vehicle objects containing all vehicles that are close to the ego vehicle
    :param ego_vehicle: vehicle object containing all necessary values from the ego vehicle in this scene (index)
    :return: integer: 1 if the current scene is "overtaking", 0 if it's not
    """
    for vehicle in relevant_vehicles:
        if (vehicle.dv > 0 and
                0 < vehicle.ds < ego_vehicle.s_0 and
                vehicle.s_road < ego_vehicle.s_road and
                ego_vehicle.lane_id == vehicle.lane_id - 1):
            return 1
    return 0


def lane_change_left_fn(data, metadata, index):
    """
    This method determines if the current scene is "lane_change_left"

    :param data: array with all signal data from that TestRun
    :param metadata: array with names of each signal
    :param index: current row in the data array, indicating the current scene
    :return: integer: 1 if the current scene is "lane_change_left", 0 if it's not
    """
    rows, columns = data.shape
    lower_index = max(index - 10, 0)
    upper_index = min(index + 10, rows - 1)
    first_lane = data[lower_index, metadata.index("Car.Road.Lane.Act.LaneId")]
    for i in range(lower_index, upper_index):
        second_lane = data[i, metadata.index("Car.Road.Lane.Act.LaneId")]
        if first_lane == second_lane + 1:
            return 1
    return 0


def lane_change_right_fn(data, metadata, index):
    """
    This method determines if the current scene is "lane_change_right"

    :param data: array with all signal data from that TestRun
    :param metadata: array with names of each signal
    :param index: current row in the data array, indicating the current scene
    :return: integer: 1 if the current scene is "lane_change_right", 0 if it's not
    """
    rows, columns = data.shape
    lower_index = max(index - 10, 0)
    upper_index = min(index + 10, rows - 1)
    first_lane = data[lower_index, metadata.index("Car.Road.Lane.Act.LaneId")]
    for i in range(lower_index, upper_index):
        second_lane = data[i, metadata.index("Car.Road.Lane.Act.LaneId")]
        if first_lane == second_lane - 1:
            return 1
    return 0


def unknown_fn(labels):
    """
    This method determines if the current scene is "unknown". It is "unknown" if no other scene is determined

    :param labels: array that indicates what scenes are determined in the current timestep (index)
    :return: integer: 1 if the current scene is "unknown", 0 if it's not
    """
    if np.sum(labels) == 0:
        return 1
    return 0


def smoothing_fn(scenes, min_consecutive_scenes):
    """
    This method verifies if the minimum number of required consecutive scenes is satisfied and returns an array
    indicating which timestep belongs to which scenario

    :param scenes: array that indicated what scenes are determined at all timesteps
    :param min_consecutive_scenes: minimum number of required consecutive scenes to be a scenario
    :return: array that indicated what scenarios are determined in all timesteps
    """
    rows, columns = scenes.shape
    scenarios = np.zeros((rows, columns))
    for i in range(columns):
        flag = 0
        for j in range(rows):
            if scenes[j, i] == 0:
                if j - flag >= min_consecutive_scenes:
                    for k in range(flag, j):
                        scenarios[k, i] = 1
                flag = j+1
    return scenarios
