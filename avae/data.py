import os
import random

import mrcfile
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from .vis import format


def load_affinities(datapath: str) -> pd.DataFrame:
    """Load affinity matrix.

    Parameters
    ----------
    datapath : str
        Path to directory containing affinity matrix denoted as '*scores*.csv'.

    Returns
    -------
    affinities : pd.DataFrame
        A square symmetric matrix where each column and row is the index of an
        object class from the training set,
        consisting of M different classes. First row and column contain IDs of
        the classes.

    """

    lookup = [
        f for f in os.listdir(datapath) if "affinity" in f and ".csv" in f
    ]

    if len(lookup) > 1:
        raise RuntimeError(
            "More than 1 affinity matrix in the root directory {}.".format(
                datapath
            )
        )
    elif not (len(lookup) == 0):
        lookup = lookup[0]
        lookup = pd.read_csv(os.path.join(datapath, lookup), header=0)
    else:
        lookup = None

    return lookup


def load_data(
    datapath: str,
    lim: int = None,
    splt: int = 20,
    batch_s: int = 64,
    no_val_drop: bool = False,
    collect_meta: bool = False,
    eval: bool = True,
):
    if not eval:
        # load affinity matrix
        lookup = load_affinities(datapath)

        # create ProteinDataset
        data = ProteinDataset(
            datapath, amatrix=lookup, lim=lim, collect_m=collect_meta
        )
        print("\nData size:", len(data))
        # dsize = data[0].shape[:-3]   # first sample, first tuple id (data)
        # print(dsize)
        # exit(1)

        # split into train / val sets
        idx = np.random.permutation(len(data))
        s = int(np.ceil(len(data) * int(splt) / 100))
        if s < 2:
            raise RuntimeError(
                "Train and validation sets must be larger than 1 sample, "
                "train: {}, val: {}.".format(len(idx[:-s]), len(idx[-s:]))
            )
        train_data = Subset(data, indices=idx[:-s])
        val_data = Subset(data, indices=idx[-s:])
        print("Train / val split:", len(train_data), len(val_data))

        # split into batches
        trains = DataLoader(
            train_data,
            batch_size=batch_s,
            num_workers=2,
            shuffle=True,
            drop_last=True,
        )
        vals = DataLoader(
            val_data,
            batch_size=batch_s,
            num_workers=2,
            shuffle=True,
            drop_last=(not no_val_drop),
        )
        tests = []
        if len(vals) < 1 or len(trains) < 1:
            # ensure the batch size is not smaller than validation set
            raise RuntimeError(
                "Validation or train set is too small for the current batch "
                "size. Please edit either split percent '-sp/--split' or batch"
                " size '-ba/--batch' or set '-nd/--no_val_drop flag' (only if "
                "val is too small). Batch: {}, train:{}, val: {}, "
                "split: {}%.".format(
                    batch_s, len(train_data), len(val_data), splt
                )
            )
        print("Train / val batches:", len(trains), len(vals))
        print()

        lookup = lookup.to_numpy(dtype=np.float32)

    if eval or ("test" in os.listdir(datapath)):
        data = ProteinDataset(
            os.path.join(datapath, "test"), lim=lim, collect_m=collect_meta
        )
        print("Eval data size:", len(data))
        tests = DataLoader(
            data, batch_size=batch_s, num_workers=2, shuffle=True
        )
        print("Eval batches:", len(tests))
        print()

    if eval:
        return tests
    else:
        return trains, vals, tests, lookup  # , dsize


class ProteinDataset(Dataset):
    """Protein dataset. Opens MRC files and returns images along with their
    affinity and associated metadata.

    Parameters
    ----------
    root_dir : string
        Base directory containing .mrc files.
    amatrix : pd.DataFrame
        A square symmetric matrix where each column and row is the index of an
        object class from the training set,
        consisting of M different classes. First row and column contain IDs of
        the classes.
    transform: torchvision.transforms.Transform
        List of transforms to be applied to the images.
    lim : int
        Limit the dataset size to the given number; useful for debugging
        purposes.
    """

    def __init__(
        self, root_dir, amatrix=None, transform=None, lim=None, collect_m=False
    ):
        super().__init__()

        self.collect_meta = collect_m

        self.amatrix = amatrix

        self.root_dir = root_dir
        self.paths = [f for f in os.listdir(root_dir) if ".mrc" in f]
        random.shuffle(self.paths)
        ids = np.unique([f.split("_")[0] for f in self.paths])

        if self.amatrix is not None:
            class_check = np.in1d(ids, self.amatrix.columns)
            if not np.all(class_check):
                raise RuntimeError(
                    "Not all classes in the training set are present in the "
                    "affinity matrix. Missing classes: {}".format(
                        np.asarray(ids)[~class_check]
                    )
                )

        if "classes.csv" in os.listdir(self.root_dir):
            with open(
                os.path.join(self.root_dir, "classes.csv"), newline="\n"
            ) as f:
                class_list = np.asarray(f.readlines()).flatten()
                class_list = [
                    c.strip() for c in class_list if len(c.strip()) != 0
                ]

            self.paths = [p for p in self.paths for c in class_list if c in p]

        self.paths = self.paths[:lim]

        if not transform:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.unsqueeze(0))
                    # transforms.Resize(64),
                    # transforms.Lambda(lambda x: \
                    # (x - x.min()) / (x.max() - x.min()))
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        """Load and image, its metadata and optionally, its affinity class
        index.

        Parameters
        ----------
        item : int
            Index of the element in the iterable.

        Returns
        -------
        data : torch.Tensor (N,)
            Transformed MRC data.
        aff : int
            Index of object's image class corresponding to rows in affinity
            matrix.
        meta : dict
            Associated metadata.
        """
        # data
        filename = self.paths[item]
        with mrcfile.open(os.path.join(self.root_dir, filename)) as f:
            data = np.array(f.data)
        x = self.transform(data)

        # ground truth
        y = filename.split("_")[0]

        # similarity column / vector
        if self.amatrix is not None:
            aff = self.amatrix.columns.get_loc(f"{y}")
        else:
            # in evaluation mode - test set
            aff = 0  # cannot be None, but not used anywhere during evaluation

        if self.collect_meta:
            # file info and metadata
            meta = "_".join(filename.split(".")[0].split("_")[1:])
            avg = np.around(np.average(x), decimals=4)
            img = format(x)  # used for dynamic preview in Altair
            meta = {
                "filename": filename,
                "id": y,
                "meta": meta,
                "avg": avg,
                "image": img,
            }

            return x, y, aff, meta
        else:
            return x, y, aff
