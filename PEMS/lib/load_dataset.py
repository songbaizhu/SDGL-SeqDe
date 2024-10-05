import os
import numpy as np
import pandas as pd


def load_st_dataset(dataset, data_path):
    # output B, N, D
    if dataset == 'PEMS04':
        # data_path = os.path.join('data/PeMSD4/pems04.npz')
        data = np.load(data_path)['data'][:, :, 0]  # onley the first dimension, traffic flow data
    elif dataset == 'PEMS08':
        # data_path = os.path.join('data/PeMSD8/pems08.npz')
        data = np.load(data_path)['data'][:, :, 0]  # onley the first dimension, traffic flow data
    elif dataset == 'PEMS03':
        # data_path = os.path.join('data/PeMSD3/PEMS03.npz')
        data = np.load(data_path)['data'][:, :, 0]
    elif dataset == 'PEMS07':
        # data_path = os.path.join('data/PeMSD7/PEMS07.npz')
        data = np.load(data_path)['data'][:, :, 0]
    elif dataset == 'METR-LA':
        data = pd.read_hdf(data_path).values
    elif dataset == 'PEMS-BAY':
        data = pd.read_hdf(data_path).values
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data
