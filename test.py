
"""
best_model = keras.models.load_model("output/model-improvement-12-1.00.hdf5")


pred_sim = []
for path in test_sim:
    pred_sim.append(my_generator.get_data(best_model, path, cnn_name, classification))
pred_sim = np.squeeze(np.array(pred_sim))
np.save(output_directory + "/predictions_test_data_sim_best.npy", pred_sim)


pred_real = []
for path in test_real:
    pred_real.append(my_generator.get_data(best_model, path, cnn_name, classification))
pred_real = np.squeeze(np.array(pred_real))
np.save(output_directory + "/predictions_test_data_real_best.npy", pred_real)
"""
