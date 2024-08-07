import copy
import logging
import os.path
import typing

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import numpy.typing as npt
import torch
from sklearn import metrics, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline


def accuracy(
    x_train: npt.NDArray,
    y_train: npt.NDArray,
    x_val: npt.NDArray,
    y_val: npt.NDArray,
    classifier: str = "NN",
) -> tuple[float, float, float, npt.NDArray, npt.NDArray]:
    """Computes the accuracy using a given classifier. Currently only supports
    neural network, K-nearest neighbors and logistic regression. A grid search on the
    hyperparameters is performed.

    Parameters
    ----------
    x_train: np.array
        Training data.
    y_train: np.array
        Training labels.
    x_val: np.array
        Validation data.
    y_val: np.array
        Validation labels.
    classifier: str
        Classifier to use. Either 'NN' for neural network, 'KNN' for K-nearest neighbors or LR for logistic regression.


    Returns
    -------
    train_acc: float
        Training accuracy.
    val_acc: float
        Validation accuracy.
    val_acc_selected: float
        Validation accuracy calculated only for labels existing in the training set (this is useful in evaluation).
    y_pred_train: np.array
        Predicted training labels.
    y_pred_val: np.array
        Predicted validation labels.

    """
    logging.info(
        "############################################### Computing accuracy..."
    )
    labs = np.unique(np.concatenate((y_train, y_val)))
    le = preprocessing.LabelEncoder()
    le.fit(labs)

    classes_list_training = np.unique(y_train)
    if np.setdiff1d(classes_list_training, np.unique(y_val)).size > 0:
        logging.info(
            f"Class {np.setdiff1d(classes_list_training, np.unique(y_val))}  was unseen in training data. Computing accuracy for sets of seen and unseen data"
        )

    index = np.argwhere(np.isin(y_val, classes_list_training)).ravel()

    y_train = le.transform(y_train)
    y_val = le.transform(y_val)

    parameters: dict[str, typing.Any]

    if classifier == "NN":

        parameters = {
            "hidden_layer_sizes": [
                (250, 150, 30),
                (100, 50, 15),
                (50, 20, 10),
                (20, 10, 5),
                (200,),
                (50,),
            ],
        }
        method = MLPClassifier(
            max_iter=500,
            activation="relu",
            solver="adam",
            random_state=1,
            alpha=1,
        )

    elif classifier == "KNN":
        parameters = dict(n_neighbors=(range(1, 500, 100)))
        method = KNeighborsClassifier()
    else:
        raise ValueError("Invalid classifier type must be NN, KNN or LR")

    clf_cv = GridSearchCV(
        estimator=method,
        param_grid=parameters,
        scoring="f1_macro",
        cv=2,
        verbose=0,
    )
    clf = make_pipeline(preprocessing.StandardScaler(), clf_cv)
    clf.fit(x_train, y_train)
    logging.info(
        f"Best parameters found for {classifier}: {clf_cv.best_params_}"
    )

    y_pred_train = clf.predict(x_train)
    y_pred_val = clf.predict(x_val)

    train_acc = metrics.accuracy_score(y_train, y_pred_train)
    val_acc = metrics.accuracy_score(y_val, y_pred_val)
    val_acc_selected = metrics.accuracy_score(
        np.array(y_val)[index].tolist(), np.array(y_pred_val)[index].tolist()
    )

    y_pred_train = le.inverse_transform(y_pred_train)
    y_pred_val = le.inverse_transform(y_pred_val)

    return train_acc, val_acc, val_acc_selected, y_pred_train, y_pred_val


def create_grid_for_plotting(
    rows: int, columns: int, dsize: tuple, padding: int = 0
) -> npt.NDArray:

    # define the dimensions for the napari grid

    if len(dsize) == 3:
        grid_for_napari = np.zeros(
            (
                rows * dsize[0],
                dsize[1] * columns + padding * columns,
                dsize[2],
            ),
            dtype=np.float32,
        )

    elif len(dsize) == 2:
        grid_for_napari = np.zeros(
            (
                rows * dsize[0],
                dsize[1] * columns + padding * columns,
            ),
            dtype=np.float32,
        )

    return grid_for_napari


def fill_grid_for_plottting(
    rows: int,
    columns: int,
    grid: npt.NDArray,
    dsize: tuple,
    array: npt.NDArray,
    padding: int = 0,
) -> npt.NDArray:

    if len(dsize) == 3:
        for j in range(columns):
            for i in range(rows):
                grid[
                    i * dsize[0] : (i + 1) * dsize[0],
                    j * (dsize[1] + padding) : (j + 1) * dsize[1]
                    + padding * j,
                    :,
                ] = array[i, j, :, :, :]

    elif len(dsize) == 2:
        for j in range(columns):
            for i in range(rows):
                grid[
                    i * dsize[0] : (i + 1) * dsize[0],
                    j * (dsize[1] + padding) : (j + 1) * dsize[1]
                    + padding * j,
                ] = array[i, j, :, :]
    return grid


def save_imshow_png(
    fname: str,
    array: npt.NDArray,
    cmap: str | None = None,
    min: float | None = None,
    max: float | None = None,
    writer: typing.Any = None,
    figname: str | None = None,
    epoch: int = 0,
    display: bool = False,
) -> None:
    if not display:
        if not os.path.exists("plots"):
            os.mkdir("plots")

        fig, _ = plt.subplots(figsize=(10, 10))
        plt.imshow(array, cmap=cmap, vmin=min, vmax=max)  # channels last

        plt.savefig("plots/" + fname)

        if not os.path.exists("plots/reconstructions"):
            os.mkdir("plots/reconstructions")
        plt.savefig("plots/reconstructions/epoch_" + str(epoch) + "_" + fname)

        if writer:
            writer.add_figure(figname, fig, epoch)
    else:
        plt.imshow(array, cmap=cmap, vmin=min, vmax=max)  # channels last
        plt.show()

    plt.close()


def save_mrc_file(fname: str, array: npt.NDArray) -> None:
    if not os.path.exists("plots"):
        os.mkdir("plots")
    with mrcfile.new("plots/" + fname, overwrite=True) as mrc:
        mrc.set_data(array)


def colour_per_class(classes: list) -> list:
    # Define the number of colors you want
    num_colors = len(classes)

    # Choose colormaps for combining
    cmap_1 = plt.get_cmap("tab20")
    cmap_2 = plt.get_cmap("Accent")
    cmap_3 = plt.get_cmap("Pastel1")
    cmap_4 = plt.get_cmap("Set1")

    # Combine the four colormaps
    combined_cmap = [cmap_1(i % 20) for i in range(20)]
    combined_cmap.extend([cmap_2(i % 8) for i in range(8)])
    combined_cmap.extend([cmap_3(i % 8) for i in range(8)])
    combined_cmap.extend([cmap_4(i % 8) for i in range(8)])

    # Create the colormap object
    custom_cmap = plt.cm.colors.ListedColormap(
        combined_cmap, name="custom_cmap"
    )

    # Generate a list of colors based on the modulo operation of i with respect to the number of colors in the combined colormap
    colours = [custom_cmap(i % len(combined_cmap)) for i in range(num_colors)]
    return colours


def pose_interpolation(
    enc: npt.NDArray,
    pos_dims: int,
    pose_mean: npt.NDArray,
    pose_std: npt.NDArray,
    dsize: tuple,
    number_of_samples: int,
    vae: torch.nn.Module,
    device: torch.device,
) -> npt.NDArray:

    """This function:
    1-  interpolates within each pose channels
        for the number_of_samples requested.
    2- returns all decoded images based on the input latent
        and the interpolated pose

    Parameters
    ----------
    enc: numpy array
        the latent encoding.
    pos_dims: int
        the pose channel dimension
    pose_mean: numpy array
        mean of each pose channel.
    pose_std: numpy array
        standard deviation of each pose channel.
    dsize: torch.size
        the dimension of the data. Example [32,32,32]
    number_of_samples: int
        number of samples to interpolate for.
    vae: torch.nn.Module
        Affinity vae model.
    device: torch.device
        Device to run the model on.
    """
    decoded_grid = []
    # Generate vectors representing single transversals along each lat_dim
    for p_dim in range(pos_dims):
        for grid_spot in range(number_of_samples):
            means = copy.deepcopy(pose_mean)
            means[p_dim] += pose_std[p_dim] * (-1.2 + 0.4 * grid_spot)

            pos = torch.from_numpy(np.array(means)).unsqueeze(0).to(device)
            lat = torch.from_numpy(np.array(enc)).unsqueeze(0).to(device)

            # Decode interpolated vectors
            with torch.no_grad():
                decoded_img, x_before_conv = vae.decoder(lat, pos)

            decoded_grid.append(decoded_img.cpu().squeeze().numpy())

    decoded_grid = np.reshape(
        np.array(decoded_grid), (pos_dims, number_of_samples, *dsize)
    )

    return decoded_grid


def latent_space_similarity_mat(
    latent_space: npt.NDArray,
    class_labels: npt.NDArray,
    unique_classes: list,
    num_classes: int,
    plot_mode: str = "",
) -> npt.NDArray:
    """
    This function calculates the similarity (affinity) between classes in the latent space and builds a matrix.
    Parameters
    ----------
    latent_space: np.ndarray
        The latent space
    class_labels: np.array
        The labels of the latent space
    mode: str
        Mode of the calculation (train, test, val)
    epoch: int
        Epoch number for title
    classes_order: list
        Order of the classes in the matrix
    display: bool
        When this variable is set to true, the function only dispalys the plot and doesnt save it.
    """
    # get same label order as affinity matrix
    cosine_sim_matrix = cosine_similarity(latent_space)

    cosine_sim_mat = np.zeros((num_classes, num_classes))

    for i in range(num_classes):
        for j in range(i, num_classes):
            class_i_indices = np.where(class_labels == unique_classes[i])[0]
            class_j_indices = np.where(class_labels == unique_classes[j])[0]
            cosine_sims = cosine_sim_matrix[class_i_indices][
                :, class_j_indices
            ]
            if plot_mode == "mean":
                cosine_sim_mat[i, j] = np.mean(cosine_sims)
                cosine_sim_mat[j, i] = cosine_sim_mat[
                    i, j
                ]  # symmetrical matrix
            if plot_mode == "std":
                cosine_sim_mat[i, j] = np.std(cosine_sims)
                cosine_sim_mat[j, i] = cosine_sim_mat[
                    i, j
                ]  # symmetrical matrix

    return cosine_sim_mat


def latest_file(path: str, extension: str) -> str:

    most_recent_file = ""
    most_recent_time = 0

    # iterate over the files in the directory using os.scandir
    for entry in os.scandir(path):
        if entry.name.lower().endswith(extension):
            # get the modification time of the file using entry.stat().st_mtime_ns
            mod_time = entry.stat().st_mtime_ns
            if (
                mod_time > most_recent_time
                and "eval" not in entry.name.lower()
            ):
                # update the most recent file and its modification time
                most_recent_file = entry.name
                most_recent_time = mod_time
    return most_recent_file
