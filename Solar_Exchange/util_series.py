import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable
import math
from torch.utils.data import DataLoader


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


class MYDataset():
    def __init__(self, data, P, h, device, is_train=True):
        super(MYDataset, self).__init__()
        self.data = data
        self.P = P
        self.h = h
        self.device = device
        self.is_train = is_train

    def __len__(self):
        return self.data.shape[0] - self.P - self.h + 1

    def __getitem__(self, index):
        x = self.data[index:index + self.P, ...]
        if self.is_train == False:
            y = self.data[index + self.P + self.h - 1, :, 0]
        else:
            y = self.data[index + self.P:index + self.P + self.h, :, 0]
        return x, y


class DataLoaderS_stamp(object):
    def __init__(self, file_name, train, valid, device, horizon, window, normalize=2, args=None):
        self.args = args
        self.P = window
        self.h = horizon
        self.device = device
        # fin = open(file_name)
        with open(file_name) as fin:
            self.rawdat = np.loadtxt(fin, delimiter=',')
        # self.rawdat = self.rawdat[:500]
        self.dat = np.zeros(self.rawdat.shape)

        self.n, self.m = self.rawdat.shape
        self.normalize = 2
        self.scale = np.ones(self.m)
        self._normalized(normalize)

        data = self.dat
        data = data[:, :, np.newaxis]
        L, N, F = data.shape
        stamp_list = [data]
        if args.add_time_in_day:
            # numerical time_in_day
            time_ind = [i % args.steps_per_day / args.steps_per_day for i in range(data.shape[0])]
            time_ind = np.array(time_ind)
            time_in_day = np.tile(time_ind, [1, N, 1]).transpose((2, 1, 0))
            stamp_list.append(time_in_day)
        if args.add_day_in_week:
            day_in_week = [(i // args.steps_per_day) % 7 for i in range(data.shape[0])]
            day_in_week = np.array(day_in_week)
            day_in_week = np.tile(day_in_week, [1, N, 1]).transpose((2, 1, 0))
            stamp_list.append(day_in_week)
        self.dat = np.concatenate(stamp_list, axis=-1)
        # self.dat = torch.from_numpy(self.dat).to(device)
        self.dat = torch.tensor(self.dat, device=device, dtype=torch.float)
        self.dritribute_data = self.dat[:int(train * self.n), :, 0]

        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        self.scale = self.scale.to(device)
        # print(' slef.scale is :', self.scale)
        testy = self.test[self.P + self.h - 1:, :, 0]
        tmp = testy * self.scale  # .expand(testy.size(0), self.m)

        # self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

        # 数据集
        train_dataset = MYDataset(self.train, self.P, self.h, device, True)
        valid_dataset = MYDataset(self.valid, self.P, self.h, device, False)
        test_dataset = MYDataset(self.test, self.P, self.h, device, False)

        # num_work = 4
        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.
        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

    def _split(self, train, valid, test):
        # train_set = range(self.P + self.h - 1, train)
        # valid_set = range(train, valid)
        # test_set = range(valid, self.n)
        # self.train = self._batchify(train_set, self.h)
        # self.valid = self._batchify(valid_set, self.h)
        # self.test = self._batchify(test_set, self.h)
        # self.valid[1] = self.valid[1][:, -1, :]  # 验证集和测试集的标签只保留最后一个
        # self.test[1] = self.test[1][:, -1, :]

        index1 = 0
        index2 = train
        self.train = self.dat[index1:index2]
        index1 = train - self.P - self.h + 1
        index2 = valid
        self.valid = self.dat[index1:index2]
        index1 = valid - self.P - self.h + 1
        self.test = self.dat[index1:]
        # print(
        #     'train data shape is {}----------train label shape is {}'.format(self.train[0].shape, self.train[1].shape))
        # print(
        #     'valid data shape is {}----------valid label shape is {}'.format(self.valid[0].shape, self.valid[1].shape))
        # print('test data shape is {}----------test label shape is {}'.format(self.test[0].shape, self.test[1].shape))

    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m, 3))
        Y = torch.zeros((n, self.h, self.m))
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :, :] = torch.from_numpy(self.dat[end:idx_set[i] + 1, :, 0])
        return [X, Y]

    # def get_batches(self, inputs, batch_size, shuffle=True):
    #     length = len(inputs)    #[0,length-P-h+1) start_index
    #     if shuffle:
    #         index = torch.randperm(length-self.P-self.h+1)
    #     else:
    #         index = torch.LongTensor(range(length-self.P-self.h+1))
    #     start_idx = 0
    #     while start_idx < length-self.P-self.h+1:
    #         index1 = index[start_idx]
    #         X = inputs[excerpt]
    #         Y = targets[excerpt]
    #
    #         yield X, Y
    #         start_idx += batch_size


class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj = load_pickle(pkl_filename)
    return adj


def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse


def load_node_feature(path):
    fi = open(path)
    x = []
    for li in fi:
        li = li.strip()
        li = li.split(",")
        e = [float(t) for t in li[1:]]
        x.append(e)
    x = np.array(x)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    z = torch.tensor((x - mean) / std, dtype=torch.float)
    return z


def normal_std(x):
    return x.std() * np.sqrt((x.numel() - 1.) / (x.numel()))


def generate_metric(total_loss, total_loss_l1, n_samples, data, predict, test):
    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    return rse, rae, correlation
