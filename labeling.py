import numpy as np
import pandas as pd
import cv2
import os


DATA_HZ = 50
FPS = 5
SMOOTHING_FACTOR = 150   # minimum number of required consecutive scenes
SCENARIOS = ["FREE CRUISING", "FOLLOWING", "OVERTAKING LEFT", "V2 OVERTAKING", "UNKNOWN"]


def get_data(path):
    # Import data from path
    data = np.genfromtxt(path)
    d = np.genfromtxt(path, comments=None, dtype=str, max_rows=1)
    d = d[1:]
    # Create list for data description (metadata)
    # global metadata
    metadata = []
    for element in d:
        metadata.append(element)
    return data, metadata


def label_scenarios(data, metadata):
    rows, columns = data.shape
    scenes_labeled = np.zeros((rows, SCENARIOS.__len__()))
    scenes_labeled[:, 0] = free_cruising_fn(data, metadata)
    scenes_labeled[:, 1] = following_fn(data, metadata)
    scenes_labeled[:, 2] = overtaking_fn(data, metadata)
    scenes_labeled[:, 3] = v2_overtaking_fn(data, metadata)
    scenes_labeled[:, 4] = unknown_fn(scenes_labeled)
    scenarios_labeled = smoothing_fn(scenes_labeled)
    scenarios_labeled[:, 4] = unknown_fn(scenarios_labeled)
    return scenes_labeled, scenarios_labeled


def get_ego_vehicle_data(data, metadata, index):
    s_0 = data[index, metadata.index("Car.v")] * 3.6
    s_1 = data[index, metadata.index("Car.v")] * 3.6 * 2 / 3
    s_2 = data[index, metadata.index("Car.v")] * 3.6 * 1 / 2
    s_3 = data[index, metadata.index("Car.v")] * 3.6 * 1 / 3
    ego_v = data[index, metadata.index("Car.v")]
    return s_0, s_1, s_2, s_3, ego_v


def free_cruising_fn(data, metadata):
    rows, columns = data.shape
    free_cruising_scenes = np.zeros((rows,))
    for i in range(rows):
        s_0, s_1, s_2, s_3, ego_v = get_ego_vehicle_data(data, metadata, i)
        if ((data[i, metadata.index("Sensor.Object.FRONT.relvTgt.NearPnt.ds_p")] > s_0 or
                data[i, metadata.index("Sensor.Object.FRONT.relvTgt.NearPnt.ds_p")] == 0) and
                data[i, metadata.index("Sensor.Object.LEFT.relvTgt.dtct")] == 0.0 and
                data[i, metadata.index("Sensor.Object.RIGHT.relvTgt.dtct")] == 0.0):
            free_cruising_scenes[i, ] = 1
    return free_cruising_scenes


def following_fn(data, metadata):
    rows, columns = data.shape
    following_scenes = np.zeros((rows,))
    for i in range(rows):
        s_0, s_1, s_2, s_3, ego_v = get_ego_vehicle_data(data, metadata, i)
        steer_ang = True
        lower_index = max(i - 50, 0)
        if lower_index == 0:
            ego_v_mean = data[i, metadata.index("Car.v")]
        else:
            ego_v_mean = data[lower_index:i, metadata.index("Car.v")].mean()
            min_ang = np.min(data[lower_index:i, metadata.index("VC.Steer.Ang")])
            max_ang = np.max(data[lower_index:i, metadata.index("VC.Steer.Ang")])
            if min_ang*1.3 < max_ang:
                steer_ang = False
        approaching = (ego_v < ego_v_mean) and \
                      (s_2 < data[i, metadata.index("Sensor.Object.FRONT.relvTgt.NearPnt.ds_p")] < s_0)
        following = (data[i, metadata.index("Sensor.Object.FRONT.relvTgt.NearPnt.dv_p")] < ego_v * 0.05) and \
                    (s_3 < data[i, metadata.index("Sensor.Object.FRONT.relvTgt.NearPnt.ds_p")] < s_1)
        if (approaching or following) and \
                abs(data[i, metadata.index("Sensor.Object.FRONT.relvTgt.NearPnt.ds.y")]) < 2 and \
                steer_ang:
            following_scenes[i, ] = 1
    return following_scenes


def overtaking_fn(data, metadata):
    rows, columns = data.shape
    overtaking_scenes = np.zeros((rows,))
    for i in range(rows):
        if data[i, metadata.index("Sensor.Object.RIGHT.relvTgt.NearPnt.dv.y")] < 0:
            overtaking_scenes[i, ] = 1
    return overtaking_scenes


def v2_overtaking_fn(data, metadata):
    rows, columns = data.shape
    v2_overtaking_scenes = np.zeros((rows,))
    for i in range(rows):
        if data[i, metadata.index("Sensor.Object.LEFT.relvTgt.NearPnt.dv.y")] < 0:
            v2_overtaking_scenes[i, ] = 1
    return v2_overtaking_scenes


def unknown_fn(scenes_labeled):
    rows, columns = scenes_labeled.shape
    unknown_scenes = np.zeros((rows,))
    for i in range(rows):
        if np.sum(scenes_labeled[i, :]) == 0:
            unknown_scenes[i, ] = 1
    return unknown_scenes


def smoothing_fn(scenes):
    rows, columns = scenes.shape
    scenarios = np.zeros((rows, columns))
    for i in range(rows):
        ix = i * SMOOTHING_FACTOR
        if ix+SMOOTHING_FACTOR >= rows:
            break
        for j in range(columns):
            if np.sum(scenes[ix:ix+SMOOTHING_FACTOR, j]) < SMOOTHING_FACTOR*0.95:
                for k in range(ix, ix+SMOOTHING_FACTOR):
                    scenarios[k, j] = 0
            else:
                for k in range(ix, ix+SMOOTHING_FACTOR):
                    scenarios[k, j] = 1
    return scenarios


def save_data(data_array, file_name):
    df = pd.DataFrame(data_array)
    df.to_csv(file_name)
    pass


def save_video(frames_folder_path, video_path, scenes_labeled, scenarios_labeled):
    number_files = len(os.listdir(frames_folder_path))
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc("X", "V", "I", "D"), float(FPS), (652, 449))
    for i in range(number_files-1):
        if i < 10:
            frame = cv2.imread(frames_folder_path + "frame-000" + str(i) + ".jpg")
        if 10 <= i < 100:
            frame = cv2.imread(frames_folder_path + "frame-00" + str(i) + ".jpg")
        if 100 <= i < 1000:
            frame = cv2.imread(frames_folder_path + "frame-0" + str(i) + ".jpg")
        if 1000 <= i < 10000:
            frame = cv2.imread(frames_folder_path + "frame-" + str(i) + ".jpg")
        if 10000 <= i < 100000:
            frame = cv2.imread(frames_folder_path + "frame-" + str(i) + ".jpg")
        video_index = i * DATA_HZ / FPS
        rows, columns = scenes_labeled.shape
        if video_index > rows:
            break
        text_scenes = ""
        for j in range(SCENARIOS.__len__()):
            if int(scenes_labeled[int(video_index), j]) == 1:
                text_scenes = text_scenes + " " + str(SCENARIOS.__getitem__(j))
        text_scenarios = ""
        for j in range(SCENARIOS.__len__()):
            if int(scenarios_labeled[int(video_index), j]) == 1:
                text_scenarios = text_scenarios + " " + SCENARIOS.__getitem__(j)
        cv2.putText(frame, text_scenarios, (150, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0))
        cv2.putText(frame, text_scenes, (150, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        cv2.imshow("title", frame)
        cv2.waitKey(1)
        out.write(frame)
    out.release()
    cv2.destroyAllWindows()
    pass
