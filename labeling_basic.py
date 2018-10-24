import numpy as np
import glob
import cv2
import my_vehicle


DATA_HZ = 50
FPS = 5
MIN_CONSECUTIVE_SCENES = 15    # minimum number of required consecutive scenes to be a scenario
SCENARIOS = ["FREE_CRUISING", "APPROACHING", "FOLLOWING",
             "CATCHING_UP", "OVERTAKING", "LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT",
             "V2_CATCHING_UP", "V2_OVERTAKING", "UNKNOWN"]


def get_data(data_path, frames_path):
    """ Import data from path """
    print("Loading data...")
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
    print("Labeling data...")
    scenes_labels = np.zeros((images.__len__(), SCENARIOS.__len__()))
    for i, image_path in enumerate(images):
        if i % 100 == 0:
            print("Status: " + str(i) + "/" + str(images.__len__()))
        data_index = int(i * DATA_HZ / FPS)
        ego_vehicle = get_ego_vehicle(data, metadata, data_index)
        relevant_vehicles = get_relevant_vehicles(data[data_index, :], metadata, all_vehicles, ego_vehicle)

        scenes_labels[i, 0] = free_cruising_fn(data[data_index, :], metadata, relevant_vehicles)
        scenes_labels[i, 1] = approaching_fn(data, metadata, relevant_vehicles, ego_vehicle, data_index)
        scenes_labels[i, 2] = following_fn(relevant_vehicles, ego_vehicle)
        scenes_labels[i, 3] = catching_up_fn(relevant_vehicles, ego_vehicle)
        scenes_labels[i, 4] = overtaking_fn(relevant_vehicles, ego_vehicle)
        scenes_labels[i, 5] = lane_change_left_fn(data, metadata, data_index)
        scenes_labels[i, 6] = lane_change_right_fn(data, metadata, data_index)
        scenes_labels[i, 7] = v2_catching_up_fn(relevant_vehicles, ego_vehicle)
        scenes_labels[i, 8] = v2_overtaking_fn(relevant_vehicles, ego_vehicle)
        scenes_labels[i, 9] = unknown_fn(scenes_labels[i, :])

    scenarios_labels = smoothing_fn(scenes_labels)
    label_dict = dict()
    for j, image_path in enumerate(images):
        scenarios_labels[j, 9] = unknown_fn(scenarios_labels[j, :])
        label_dict.update({image_path: scenarios_labels[j, :]})
    return label_dict, SCENARIOS


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
        return 1
    return 0


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
            return 1
    return 0


def following_fn(relevant_vehicles, ego_vehicle):
    for vehicle in relevant_vehicles:
        if (vehicle.dv < ego_vehicle.v * 0.05 and
                ego_vehicle.s_3 < vehicle.ds < ego_vehicle.s_1 and
                ego_vehicle.s_road < vehicle.s_road and
                ego_vehicle.lane_id == vehicle.lane_id):
            return 1
    return 0


def catching_up_fn(relevant_vehicles, ego_vehicle):
    for vehicle in relevant_vehicles:
        if (vehicle.dv < 0 and
                0 <= vehicle.ds < ego_vehicle.s_0 and
                ego_vehicle.s_road <= vehicle.s_road and
                ego_vehicle.lane_id == vehicle.lane_id - 1):
            return 1
    return 0


def overtaking_fn(relevant_vehicles, ego_vehicle):
    for vehicle in relevant_vehicles:
        if (vehicle.dv > 0 and
                0 < vehicle.ds < ego_vehicle.s_0 and
                vehicle.s_road < ego_vehicle.s_road and
                ego_vehicle.lane_id == vehicle.lane_id - 1):
            return 1
    return 0


def lane_change_left_fn(data, metadata, index):
    rows, columns = data.shape
    lower_index = max(index - 200, 0)
    upper_index = min(index + 200, rows - 1)
    first_lane = data[lower_index, metadata.index("Car.Road.Lane.Act.LaneId")]
    for i in range(lower_index, upper_index):
        second_lane = data[i, metadata.index("Car.Road.Lane.Act.LaneId")]
        if first_lane == second_lane + 1:
            return 1
    return 0


def lane_change_right_fn(data, metadata, index):
    rows, columns = data.shape
    lower_index = max(index - 200, 0)
    upper_index = min(index + 200, rows - 1)
    first_lane = data[lower_index, metadata.index("Car.Road.Lane.Act.LaneId")]
    for i in range(lower_index, upper_index):
        second_lane = data[i, metadata.index("Car.Road.Lane.Act.LaneId")]
        if first_lane == second_lane - 1:
            return 1
    return 0


def v2_catching_up_fn(relevant_vehicles, ego_vehicle):
    for vehicle in relevant_vehicles:
        s_0_v2 = (vehicle.dv + ego_vehicle.v) * 3.6
        if (0 < vehicle.dv and
                0 <= vehicle.ds < s_0_v2 and
                vehicle.s_road <= ego_vehicle.s_road and
                vehicle.lane_id == ego_vehicle.lane_id - 1):
            return 1
    return 0


def v2_overtaking_fn(relevant_vehicles, ego_vehicle):
    for vehicle in relevant_vehicles:
        s_0_v2 = (vehicle.dv + ego_vehicle.v) * 3.6
        if (0 < vehicle.dv and
                0 < vehicle.ds < s_0_v2 and
                ego_vehicle.s_road < vehicle.s_road and
                vehicle.lane_id == ego_vehicle.lane_id - 1):
            return 1
    return 0


def unknown_fn(labels):
    if np.sum(labels) == 0:
        return 1
    return 0


def smoothing_fn(scenes):
    rows, columns = scenes.shape
    scenarios = np.zeros((rows, columns))
    for i in range(columns):
        flag = 0
        for j in range(rows):
            if scenes[j, i] == 0:
                if j - flag >= MIN_CONSECUTIVE_SCENES:
                    for k in range(flag, j):
                        scenarios[k, i] = 1
                flag = j+1
    return scenarios


def save_video(label_dict, video_path):
    print("Saving video...")
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc("X", "V", "I", "D"), float(FPS), (652, 449))
    for image_path in label_dict:
        frame = cv2.imread(image_path)
        text_scenarios = ""
        for j in range(SCENARIOS.__len__()):
            if label_dict[image_path][j] == 1:
                text_scenarios = text_scenarios + " " + SCENARIOS[j]
        # todo text for scenes
        cv2.putText(frame, text_scenarios, (150, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0))
        # cv2.putText(frame, text_scenes, (150, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0))
        # cv2.imshow("title", frame)
        # cv2.waitKey(1)
        out.write(frame)
    out.release()
    cv2.destroyAllWindows()
    pass

