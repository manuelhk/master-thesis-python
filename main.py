import labeling
import time

dir = "/Users/manuel/Dropbox/_data/"

print("Loading data...")

start = time.clock()
d, m = labeling.get_data(dir + "data/0810/hc/data.dat")
diff = time.clock() - start

print("Loading data took " + str(round(diff, 2)) + " seconds")
print("Labeling data...")

start = time.clock()
scenes_labeled, scenarios_labeled = labeling.label_scenarios(d, m)
diff = time.clock() - start

print("Labeling data took " + str(round(diff, 2)) + " seconds")
print("Saving video...")

start = time.clock()
labeling.save_video(dir + "data/0810/hc/frames/", dir + "data/0810/hc/video.avi", scenes_labeled, scenarios_labeled)
diff = time.clock() - start

print("Saving video took " + str(round(diff, 2)) + " seconds")
