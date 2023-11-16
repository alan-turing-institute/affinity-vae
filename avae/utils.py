import logging
import os.path

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import yaml
from sklearn import metrics, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

from . import config


def accuracy(x_train, y_train, x_val, y_val, classifier="NN"):
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

    elif classifier == "LR":
        method = LogisticRegression(
            random_state=0, multi_class="multinomial", max_iter=1000
        )

        parameters = [{"penalty": ["l1", "l2"]}, {"C": [1, 10, 100, 1000]}]

    elif classifier == "KNN":
        parameters = dict(n_neighbors=list(range(1, 500, 100)))
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


def create_grid_for_plotting(rows, columns, dsize, padding=0):

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


def fill_grid_for_plottting(rows, columns, grid, dsize, array, padding=0):

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
    fname,
    array,
    cmap=None,
    min=None,
    max=None,
    writer=None,
    figname=None,
    epoch=0,
):
    if not os.path.exists("plots"):
        os.mkdir("plots")

    fig, _ = plt.subplots(figsize=(10, 10))
    plt.imshow(array, cmap=cmap, vmin=min, vmax=max)  # channels last
    plt.axis("off")

    plt.savefig("plots/" + fname, bbox_inches="tight", pad_inches=0)

    if writer:
        writer.add_figure(figname, fig, epoch)

    plt.close()


def save_mrc_file(fname, array):
    if not os.path.exists("plots"):
        os.mkdir("plots")
    with mrcfile.new("plots/" + fname, overwrite=True) as mrc:
        mrc.set_data(array)


def load_config_params(config_file, local_vars):

    if config_file is not None:
        with open(config_file, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        # returns JSON object as

        for key, val in local_vars.items():
            if (
                val is not None
                and isinstance(val, (int, float, bool, str))
                or data.get(key) is None
            ):
                logging.warning(
                    "Command line argument "
                    + key
                    + " is overwriting config file value to: "
                    + str(val)
                )
                data[key] = val
            else:
                logging.info(
                    "Setting "
                    + key
                    + " to config file value: "
                    + str(data[key])
                )
    else:
        # if no config file is provided, use command line arguments
        data = local_vars

        # Check for missing values and set to default values
    for key, val in data.items():
        if (val is None or val == "None") and key != "config_file":
            #  make sure data variables are provided
            if key == "data_path":
                logging.error(
                    "No value set for "
                    + key
                    + " in config file or command line arguments. Please set a value for this variable."
                )
                raise ValueError(
                    "No value set for "
                    + key
                    + " in config file or command line arguments. Please set a value for this variable."
                )
            elif key == "affinity" or key == "classes":
                logging.warning(
                    "No value set for "
                    + key
                    + " in config file or command line arguments. Setting to default value."
                )
                filename_default = os.path.join(data["datapath"], key + ".csv")

                if os.path.isfile(filename_default):
                    data[key] = filename_default
                else:
                    data[key] = None

                logging.info(
                    "Setting up "
                    + key
                    + " in config file to "
                    + str(data[key])
                )

            elif key == "state":
                logging.warning(
                    "No value set for "
                    + key
                    + " in config file or command line arguments. Loading the latest state if in evaluation mode."
                )
            elif key == "meta":
                logging.warning(
                    "No value set for "
                    + key
                    + " in config file or command line arguments. Loading the latest meta if in evaluation mode."
                )
            else:
                # set missing variables to default value
                logging.warning(
                    "No value set for "
                    + key
                    + " in config file or command line arguments. Setting to default value."
                )
                data[key] = config.DEFAULT_RUN_CONFIGS[key]
                logging.info(
                    "Setting " + key + " to default value: " + str(data[key])
                )

    return data


def write_config_file(time_stamp_name, data):
    # record final configuration in logger and save to yaml file
    for key, val in data.items():
        logging.info("Parameter " + key + " set to value: " + str(data[key]))

    if not os.path.exists("configs"):
        os.mkdir("configs")
    file = open("configs/avae_final_config" + time_stamp_name + ".yaml", "w")
    yaml.dump(data, file)
    file.close()

    logging.info("YAML File saved!\n")


def colour_per_class(classes: list):
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
