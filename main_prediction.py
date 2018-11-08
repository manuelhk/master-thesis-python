import keras
import prediction
import glob


labels = ["free_cruising", "following", "catching_up", "lane_change_left", "lane_change_right"]
scenario_paths = glob.glob("data/video/*.npy")

model = keras.models.load_model("output/1107_v3_lstm_fr_fo_ca_lcl_lcr_950_50.h5")
