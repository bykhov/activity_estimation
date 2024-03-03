import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import seaborn as sns
from tqdm import tqdm

from gamma_simulator import gamma_simulator

data = np.load('gamma_shape_parameters.npz')
param1 = data['alpha_param']
param2 = data['beta_param']
(energy_bins, energy_weights) = np.load('energy_histogram.npz').values()

# %%
N = 1000
lambda_values = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.15, 0.2, 0.3]
X_train = np.zeros((N * len(lambda_values), 1024))
Y_train = np.zeros(X_train.shape)
X_test = np.zeros((N * len(lambda_values), 1024))
Y_test = np.zeros(X_test.shape)
cnt = 0
for lambda_value in lambda_values:
    simulator = gamma_simulator(verbose=False,
                                source={'hist_energy': energy_bins,
                                        'hist_counts': energy_weights},  # Energy histogram
                                lambda_value=lambda_value,
                                signal_len=1024,
                                dict_type='gamma',
                                dict_shape_params={'custom': True,
                                                   'param1val': param1,
                                                   'param2val': param2},
                                noise=30,
                                dict_size=100)
    for _ in tqdm(range(N)):  # 1000 signals per lambda
        X_train[cnt, :] = simulator.generate_signal()
        labels = np.zeros((1024,))
        labels[simulator.times.astype(int)] = 1
        Y_train[cnt,:] = labels

        X_test[cnt, :] = simulator.generate_signal()
        labels = np.zeros((1024,))
        labels[simulator.times.astype(int)] = 1
        Y_test[cnt] = labels
        cnt += 1


#%%
np.savez('database_gamma.npz', X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
#%%
