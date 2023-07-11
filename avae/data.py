import os
import random

import mrcfile
import numpy as np
import pandas as pd
from torch import from_numpy
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from . import config
from .vis import format, plot_affinity_matrix, plot_classes_distribution

np.random.seed(42)


def load_data(
    datapath: str,
    lim: int = None,
    splt: int = 20,
    batch_s: int = 64,
    no_val_drop: bool = False,
    collect_meta: bool = False,
    eval: bool = True,
    affinity=None,
    classes=None,
    gaussian_blur=False,
    normalise=False,
    shift_min=False,
    bandpass=False,
    bp_low=None,
    bp_high=None,
):
    """Loads all data needed for training, testing and evaluation. Loads MRC files from a given path, selects subset of
    classes if requested, splits it into train / val  and test in batch sets, loads affinity matrix. Returns train,
    validation and test data as DataLoader objects.

    Parameters
    ----------
    datapath: str
        Path to the data directory.
    lim: int
        Limit the number of samples to load.
    splt: int
        Percentage of data to be used for validation.
    batch_s: int
        Batch size.
    no_val_drop: bool
        If True, the last batch of validation data will not be dropped if it is smaller than batch size.
    collect_meta: bool
        If True, the meta data for visualisation will be collected and returned.
    eval: bool
        If True, the data will be loaded only for evaluation.
    affinity: str
        Path to the affinity matrix.
    classes: list
        List of classes to be selected from the data.
    gaussian_blur: bool
        if True, Gaussian bluring is applied to the input before being passed to the model.
        This is added as a way to remove noise from the input data.
    normalise:
        In True, the input data is normalised before being passed to the model.
    shift_min: bool
        If True, the minimum value of the input data is shifted to 0 and maximum to 1.


    Returns
    -------
    train_data: DataLoader
        Train data, returned only if eval is False.
    val_data: DataLoader
        Validation data, returned only if eval is False.
    test_data: DataLoader
        Test data, if eval is True only test data is returned.
    lookup: pd.DataFrame
        Affinity matrix, returned only if eval is False.
    """

    if not eval:
        if affinity is not None:
            # load affinity matrix
            lookup = pd.read_csv(affinity, header=0)
        else:
            lookup = None

        # create ProteinDataset
        data = ProteinDataset(
            datapath,
            amatrix=lookup,
            classes=classes,
            gaussian_blur=gaussian_blur,
            normalise=normalise,
            shift_min=shift_min,
            bandpass=bandpass,
            bp_low=bp_low,
            bp_high=bp_high,
            lim=lim,
            collect_m=collect_meta,
        )
        print("\nData size:", len(data), flush=True)
        print("\nClass list:", data.final_classes, flush=True)

        if affinity is not None:
            plot_affinity_matrix(
                lookup=lookup,
                all_classes=lookup.columns.tolist(),
                selected_classes=data.final_classes,
            )

        # updating affinity matrix with the final classes
        lookup = data.amatrix

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
        print("Train / val split:", len(train_data), len(val_data), flush=True)

        # ############################### Visualising class distribution ###############################

        train_y = [y[1] for _, y in enumerate(train_data)]
        val_y = [y[1] for _, y in enumerate(val_data)]

        if config.VIS_HIS:
            plot_classes_distribution(train_y, "train")
            plot_classes_distribution(val_y, "validation")

        # split into batches
        trains = DataLoader(
            train_data,
            batch_size=batch_s,
            num_workers=0,
            shuffle=True,
            drop_last=True,
        )
        vals = DataLoader(
            val_data,
            batch_size=batch_s,
            num_workers=0,
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
                "val is too small). Batch: {}, train: {}, val: {}, "
                "split: {}%.".format(
                    batch_s, len(train_data), len(val_data), splt
                )
            )
        print("Train / val batches:", len(trains), len(vals), flush=True)
        print(flush=True)

        if affinity is not None:
            lookup = lookup.to_numpy(dtype=np.float32)
        else:
            lookup = None

    if eval or ("test" in os.listdir(datapath)):
        if "test" in os.listdir(datapath):
            datapath = os.path.join(datapath, "test")
        data = ProteinDataset(datapath, lim=lim, collect_m=collect_meta)
        print("Eval data size:", len(data), flush=True)
        tests = DataLoader(
            data, batch_size=batch_s, num_workers=0, shuffle=True
        )
        print("Eval batches:", len(tests), flush=True)
        print(flush=True)

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
    gaussian_blur: bool
        if True, Gaussian bluring is applied to the input before being passed to the model.
        This is added as a way to remove noise from the input data.
    normalise:
        In True, the input data is normalised before being passed to the model.
    shift_min: bool
        If True, the minimum value of the input data is shifted to 0 and maximum to 1.
    lim : int
        Limit the dataset size to the given number; useful for debugging
        purposes.
    collect_meta: bool
        If True, the meta data for visualisation will be collected and returned.
    """

    def __init__(
        self,
        root_dir,
        amatrix=None,
        classes=None,
        transform=None,
        gaussian_blur=False,
        normalise=False,
        shift_min=False,
        bandpass=False,
        bp_low=None,
        bp_high=None,
        lim=None,
        collect_m=False,
    ):
        super().__init__()
        self.transform = transform
        self.gaussian_blur = gaussian_blur
        self.normalise = normalise
        self.shift_min = shift_min
        self.bandpass = bandpass
        self.bp_low = bp_low
        self.bp_high = bp_high

        if self.shift_min:
            print(
                "Data Transformation : Shift the minimum of the data to one zero and the maximum to one",
                flush=True,
            )

        if self.gaussian_blur:
            print(
                "Data Transformation : GaussianBlur is applied to the images",
                flush=True,
            )

        if self.normalise:
            print(
                "Data Transformation : Normalisation is applied to the images",
                flush=True,
            )
        if self.bandpass:
            print(
                "Data Transformation :BandPasss filter is applied to input with lower threshhold of {bp_low} and higher threshhold of {bp_high}",
                flush=True,
            )

        self.collect_meta = collect_m

        self.amatrix = amatrix

        self.root_dir = root_dir
        self.paths = [f for f in os.listdir(root_dir) if ".mrc" in f]
        random.shuffle(self.paths)
        ids = np.unique([f.split("_")[0] for f in self.paths])
        self.final_classes = ids

        if classes is not None:
            classes_list = pd.read_csv(classes).columns.tolist()
            self.final_classes = classes_list

        if self.amatrix is not None:
            class_check = np.in1d(self.final_classes, self.amatrix.columns)
            if not np.all(class_check):
                raise RuntimeError(
                    "Not all classes in the training set are present in the "
                    "affinity matrix. Missing classes: {}".format(
                        np.asarray(ids)[~class_check]
                    )
                )

            # subset affinity matrix with only the relevant classes
            index = [
                self.amatrix.columns.get_loc(f"{columns}")
                for columns in self.final_classes
            ]
            self.amatrix = self.amatrix.iloc[index, index]

        self.paths = [
            p for p in self.paths for c in self.final_classes if c in p
        ]

        self.paths = self.paths[:lim]

    def transformations(self, x):

        # convert numpy to torch tensor
        x = from_numpy(x)

        # unsqueeze adds a dimension for batch processing the data
        x = x.unsqueeze(0)

        if self.shift_min:
            print(
                "Data Transformation : Shift the minimum of the data to one zero and the maximum to one",
                flush=True,
            )
            x = (x - x.min()) / (x.max() - x.min())

        if self.gaussian_blur:
            print(
                "Data Transformation : GaussianBlur is applied to the images",
                flush=True,
            )
            T = transforms.GaussianBlur(3, sigma=(0.08, 10.0))
            x = T(x)

        if self.normalise:
            print(
                "Data Transformation : Normalisation is applied to the images",
                flush=True,
            )
            T = transforms.Normalize(0, 1, inplace=False)
            x = T(x)

        if self.bandpass:
            image = x[0, :, :, :]
            F = np.fft.fftn(image)
            Fshift = np.fft.fftshift(F)

            bandpass_filter = np.zeros((F.shape), dtype=np.float32)
            if len(F.shape) == 2:
                for u in range(F.shape[0]):
                    for v in range(F.shape[1]):
                        D = np.sqrt(
                            (u - F.shape[0] / 2) ** 2
                            + (v - F.shape[1] / 2) ** 2
                        )
                        if D <= self.bp_low:
                            bandpass_filter[u, v] = 1
                        elif D >= self.bp_high:
                            bandpass_filter[u, v] = 1
            elif len(F.shape) == 3:
                for u in range(F.shape[0]):
                    for v in range(F.shape[1]):
                        for w in range(F.shape[2]):
                            D = np.sqrt(
                                (u - F.shape[0] / 2) ** 2
                                + (v - F.shape[1] / 2) ** 2
                                + (w - F.shape[2] / 2) ** 2
                            )

                            if D <= self.bp_low:
                                bandpass_filter[u, v, w] = 1
                            elif D >= self.bp_high:
                                bandpass_filter[u, v, w] = 1

            Gshift = Fshift * bandpass_filter
            G = np.fft.ifftshift(Gshift)
            x[0, :, :, :] = from_numpy(abs(np.fft.ifftn(G)))

        if self.transform:
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        """
        Load and image, its metadata and optionally, its affinity class
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

        x = self.transformations(data)
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
