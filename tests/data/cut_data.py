import os
import numpy as np

folder = "/media/fcerino/66E7-1013/hpgreedy"

ts = np.load(os.path.join(folder, "wfs_1d_seed=1.npy"))
params = np.load(os.path.join(folder, "parameters_1d_seed=1.npy"))
times = np.load(os.path.join(folder, "times_1d_seed=1.npy"))

n_train = 500

q_train = params[:n_train]
q_test = params[n_train:n_train*2]

ts_train = ts[:n_train]
ts_test = ts[n_train:n_train*2]

del ts, params

print(q_train.shape,ts_train.shape)
print(q_test.shape,ts_test.shape)
print(times.shape)

np.save("q_train_1d_seed_eq_1",q_train)
np.save("q_test_1d_seed_eq_1",q_test)
np.save("ts_train_1d_seed_eq_1",ts_train)
np.save("ts_test_1d_seed_eq_1",ts_test)
np.save("times_1d_seed_eq_1",times)