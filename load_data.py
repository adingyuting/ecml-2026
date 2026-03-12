import os
import random
import numpy as np
import torch
import pickle as pk
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from utils import StandardScaler


def _read_txt_matrix(path):
    """Read ``uid::time::value`` formatted text into a node-time matrix."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            uid, tid, val = line.split("::")
            records.append((int(uid), int(tid), float(val)))
    if not records:
        raise ValueError(f"Empty data file: {path}")
    # convert to 0-based indices in case files start from 1
    min_uid = min(r[0] for r in records)
    min_tid = min(r[1] for r in records)
    shift_uid = min_uid if min_uid > 0 else 0
    shift_tid = min_tid if min_tid > 0 else 0
    num_nodes = max(r[0] for r in records) - shift_uid + 1
    num_times = max(r[1] for r in records) - shift_tid + 1
    data = np.full((num_nodes, num_times), np.nan, dtype=np.float32)
    for uid, tid, val in records:
        data[uid - shift_uid, tid - shift_tid] = val
    return data


def _build_sequences(matrix, seqlen):
    """Convert node-time matrix into (B,N,L,1) sequences with masks."""
    num_nodes, total_time = matrix.shape
    seqs, masks = [], []
    for t in range(total_time - seqlen + 1):
        window = matrix[:, t : t + seqlen]
        mask = ~np.isnan(window)
        seqs.append(np.nan_to_num(window)[..., None])
        masks.append(mask.astype(np.float32)[..., None])
    X = np.stack(seqs)
    M = np.stack(masks)
    Y = X.copy()
    return X, M, Y


def load_txt_data(dataset, seqlen, base_dir):
    """Load data stored in ``uid::time::value`` text files.

    ``base_dir`` points to the folder containing ``train.txt`` and ``test.txt``
    for a particular missing-data ratio.
    """
    train_mat = _read_txt_matrix(os.path.join(base_dir, "train.txt"))
    test_mat = _read_txt_matrix(os.path.join(base_dir, "test.txt"))

    train_X, train_M, train_Y = _build_sequences(train_mat, seqlen)
    test_X, test_M, test_Y = _build_sequences(test_mat, seqlen)

    # use test set as validation set
    valid_X, valid_M, valid_Y = test_X.copy(), test_M.copy(), test_Y.copy()

    observed = train_X[train_M == 1]
    mean = observed.mean()
    std = observed.std() if observed.std() > 0 else 1.0

    train_X = (train_X - mean) / std
    train_Y = (train_Y - mean) / std
    valid_X = (valid_X - mean) / std
    valid_Y = (valid_Y - mean) / std
    test_X = (test_X - mean) / std
    test_Y = (test_Y - mean) / std

    A = np.eye(train_X.shape[1], dtype=np.float32)

    return (
        train_X,
        train_M,
        train_Y,
        valid_X,
        valid_M,
        valid_Y,
        test_X,
        test_M,
        test_Y,
        A,
        mean,
        std,
    )


def load_data(dataset, miss_mechanism, miss_ratio, seqlen):
    """Read dataset in either original pkl format or custom txt format."""
    base = os.path.join("datasets", dataset, f"{miss_ratio}")
    train_txt = os.path.join(base, "train.txt")
    test_txt = os.path.join(base, "test.txt")
    if os.path.exists(train_txt) and os.path.exists(test_txt):
        return load_txt_data(dataset, seqlen, base)

    # fallback: look for txt files directly under the dataset folder
    base = os.path.join("datasets", dataset)
    train_txt = os.path.join(base, "train.txt")
    test_txt = os.path.join(base, "test.txt")
    if os.path.exists(train_txt) and os.path.exists(test_txt):
        return load_txt_data(dataset, seqlen, base)

    # default: original pkl-based pipeline
    with open("datasets/{}/adj_mx.pkl".format(dataset), "rb") as fb:
        A = pk.load(fb).astype(np.float32)
    with open("datasets/{}/processed/normalization.pkl".format(dataset), "rb") as fb:
        mean, std = pk.load(fb)["args"].values()

    with open(
        "datasets/{}/processed/{}/train_{}_ms{}_seqlen_{}.pkl".format(
            dataset, miss_mechanism, miss_mechanism, 0.0, seqlen
        ),
        "rb",
    ) as fb:
        train_data = pk.load(fb)
    train_X = np.expand_dims(np.nan_to_num(train_data["data"]), 3).astype(np.float32)
    train_M = np.expand_dims(train_data["mask"].astype(np.float32), 3)
    train_Y = np.expand_dims(train_data["target"].astype(np.float32), 3)

    with open(
        "datasets/{}/processed/{}/valid_{}_ms{}_seqlen_{}.pkl".format(
            dataset, miss_mechanism, miss_mechanism, 0.0, seqlen
        ),
        "rb",
    ) as fb:
        valid_data = pk.load(fb)
    valid_X = np.expand_dims(np.nan_to_num(valid_data["data"]), 3).astype(np.float32)
    valid_M = np.expand_dims(valid_data["mask"].astype(np.float32), 3)
    valid_Y = np.expand_dims(valid_data["target"].astype(np.float32), 3)

    with open(
        "datasets/{}/processed/{}/test_{}_ms{}_seqlen_{}.pkl".format(
            dataset, miss_mechanism, miss_mechanism, 0.0, seqlen
        ),
        "rb",
    ) as fb:
        test_data = pk.load(fb)
    test_X = np.expand_dims(np.nan_to_num(test_data["data"]), 3).astype(np.float32)
    test_M = np.expand_dims(test_data["mask"].astype(np.float32), 3)
    test_Y = np.expand_dims(test_data["target"].astype(np.float32), 3)

    return (
        train_X,
        train_M,
        train_Y,
        valid_X,
        valid_M,
        valid_Y,
        test_X,
        test_M,
        test_Y,
        A,
        mean,
        std,
    )


def generate_miss_loader(
    dataset,
    miss_mechanism,
    miss_ratio,
    seqlen,
    batch_size,
    val_batch_size,
    test_batch_size,
    selection_split="valid",
):
    def add_pos_emb(X, Y):
        B, N, L, _ = X.shape
        pos = np.arange(L).reshape(1, 1, L, 1)
        pos = np.tile(pos, (B, N, 1, 1))
        pos = pos.transpose(0, 1, 3, 2)  # (B,N,1,L)
        X = X.transpose(0, 1, 3, 2)
        Y = Y.transpose(0, 1, 3, 2)
        X = np.concatenate([X, pos], axis=2).astype(np.float32)
        Y = np.concatenate([Y, pos], axis=2).astype(np.float32)
        return X, Y

    (
        train_X,
        train_M,
        train_Y,
        valid_X,
        valid_M,
        valid_Y,
        test_X,
        test_M,
        test_Y,
        A,
        mean,
        std,
    ) = load_data(dataset, miss_mechanism, miss_ratio, seqlen)

    # train
    train_M = np.repeat(train_M, 2, axis=-1)
    train_M = train_M.transpose(0, 1, 3, 2)
    train_X, train_Y = add_pos_emb(train_X, train_Y)

    # valid uses test set
    valid_M = np.repeat(valid_M, 2, axis=-1)
    valid_M = valid_M.transpose(0, 1, 3, 2)
    valid_X, valid_Y = add_pos_emb(valid_X, valid_Y)

    # test
    test_M = np.repeat(test_M, 2, axis=-1)
    test_M = test_M.transpose(0, 1, 3, 2)
    test_X, test_Y = add_pos_emb(test_X, test_Y)

    print(
        "train X: {} train M: {} train Y: {}".format(
            train_X.shape, train_M.shape, train_Y.shape
        )
    )
    print(
        "valid X: {} valid M: {} valid Y: {}".format(
            valid_X.shape, valid_M.shape, valid_Y.shape
        )
    )
    print(
        "test X: {} test M: {} test Y: {}".format(
            test_X.shape, test_M.shape, test_Y.shape
        )
    )

    train_dataset = TensorDataset(
        torch.from_numpy(train_X), torch.from_numpy(train_M), torch.from_numpy(train_Y)
    )

    valid_dataset = TensorDataset(
        torch.from_numpy(valid_X), torch.from_numpy(valid_M), torch.from_numpy(valid_Y)
    )

    test_dataset = TensorDataset(
        torch.from_numpy(test_X), torch.from_numpy(test_M), torch.from_numpy(test_Y)
    )

    if selection_split == "test":
        selection_dataset = test_dataset
    else:
        selection_dataset = valid_dataset

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(selection_dataset, batch_size=val_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size)

    print(f"selection split: {selection_split}")

    return train_loader, valid_loader, test_loader, mean, std, A


class MissDataLoader(object):

    def __init__(
        self, xs, ms, ys, batch_size, pad_with_last_sample=True, shuffle=False
    ):
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
            m_padding = np.repeat(ms[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ms = np.concatenate([ms, m_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ms, ys = xs[permutation], ms[permutation], ys[permutation]
        self.xs = xs
        self.ms = ms
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                m_i = self.ms[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                yield (x_i, m_i, y_i)
                self.current_ind += 1

        return _wrapper()

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return torch.Tensor(self.xs[idx]), (
            torch.Tensor(self.ys[idx]),
            torch.Tensor(self.ms[idx]),
        )


def load_missdataset(
    dataset, miss_mechanism, miss_ratio, seqlen, batch_size, test_batch_size
):
    data = {}

    (
        train_X,
        train_M,
        train_Y,
        valid_X,
        valid_M,
        valid_Y,
        test_X,
        test_M,
        test_Y,
        A,
        mean,
        std,
    ) = load_data(dataset, miss_mechanism, miss_ratio, seqlen)

    train_X = train_X.transpose(0, 2, 1, 3)  # (B,L,N,1)
    train_M = train_M.transpose(0, 2, 1, 3)  # (B,L,N,1)
    train_Y = train_Y.transpose(0, 2, 1, 3)  # (B,L,N,1)

    valid_X = valid_X.transpose(0, 2, 1, 3)  # (B,L,N,1)
    valid_M = valid_M.transpose(0, 2, 1, 3)  # (B,L,N,1)
    valid_Y = valid_Y.transpose(0, 2, 1, 3)  # (B,L,N,1)

    test_X = test_X.transpose(0, 2, 1, 3)  # (B,L,N,1)
    test_M = test_M.transpose(0, 2, 1, 3)  # (B,L,N,1)
    test_Y = test_Y.transpose(0, 2, 1, 3)  # (B,L,N,1)

    scaler = StandardScaler(mean, std)
    data["m_train"] = train_M
    data["m_val"] = valid_M
    data["m_test"] = test_M
    data["y_test"] = test_Y
    data["train_loader"] = MissDataLoader(
        train_X, train_M, train_Y, batch_size, pad_with_last_sample=False, shuffle=True
    )
    data["val_loader"] = MissDataLoader(
        valid_X,
        valid_M,
        valid_Y,
        test_batch_size,
        pad_with_last_sample=False,
        shuffle=False,
    )
    data["test_loader"] = MissDataLoader(
        test_X,
        test_M,
        test_Y,
        test_batch_size,
        pad_with_last_sample=True,
        shuffle=False,
    )
    data["scaler"] = scaler
    return data