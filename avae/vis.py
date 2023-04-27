import copy
import os.path
import random

import altair
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import torch
import torchvision
import umap
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay


def _encoder(i):
    """Encode PIL Image as base64 buffer."""
    import base64
    from io import BytesIO

    with BytesIO() as buffer:
        i.thumbnail((110, 110))
        i.save(buffer, "PNG")
        data = base64.encodebytes(buffer.getvalue()).decode("utf-8")

    return f"{data}"


def _decoder(i):
    """Decode base64 buffer as PIL Image."""
    import base64
    from io import BytesIO

    return Image.open(BytesIO(base64.b64decode(i)))


def format(im):
    """Format PIL Image as Pandas compatible Altair image display."""
    if len(im.shape) == 5:
        batch = True
        im = np.sum(
            np.copy(im.squeeze(dim=1).cpu().detach().numpy()), axis=-1
        )  # .astype(np.uint8)
    elif len(im.shape) == 4:
        batch = False
        im = np.sum(
            np.copy(im.squeeze(dim=0).cpu().detach().numpy()), axis=-1
        )  # .astype(np.uint8)
    else:
        print(
            "WARNING: Wrong data format, please pass either a single "
            "unsqueezed tensor or a batch to image formatter. Exiting.\n"
        )
        return
    im *= 255
    im = im.astype(np.uint8)
    if batch:
        # if batch we are adding a reconstruction to an input that already
        # exists in the DF, '&' is a separator.
        return ["&" + _encoder(Image.fromarray(i)) for i in im]
    else:
        return _encoder(Image.fromarray(im))


def merge(im):
    """Merge 2 base64 buffers as PIL Images and encode back to base64
    buffers."""
    i = im.split("&")
    if len(i) != 2:
        print(
            "WARNING: Image format corrupt. Number of images in meta_df: {}. "
            "Exiting. \n".format(len(i))
        )
        return

    im1 = _decoder(i[0])
    im2 = _decoder(i[1])

    new_image = Image.new("L", (im2.size[0] + im1.size[0], im2.size[1]))
    new_image.paste(im1, (0, 0))
    new_image.paste(im2, (im1.size[0], 0))
    data = _encoder(new_image)

    return f"data:image/png;base64,{data}"


def latent_embed_plot(xs, ys):
    print("\n################################################################")
    print("Visualising static TSNE embedding...\n")
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    labs = np.unique(ys)
    lats = TSNE(n_components=2).fit_transform(xs)
    plt.clf()
    plt.scatter(
        lats[:, 0], lats[:, 1], c=[list(labs).index(i) for i in ys], label=labs
    )
    plt.savefig("plots/embedding.png")


def dyn_latentembed_plot(df, epoch, embedding="umap"):
    print("\n################################################################")
    print("Visualising dynamic embedding {}...\n".format(embedding))

    epoch += 1
    latentspace = df[[col for col in df if col.startswith("lat")]].to_numpy()
    if embedding == "umap":
        lat_emb = np.array(umap.UMAP().fit_transform(latentspace))
        titlex = "UMAP-1"
        titley = "UMAP-2"
    else:
        lat_emb = np.array(TSNE(n_components=2).fit_transform(latentspace))
        titlex = "t-SNE-1"
        titley = "t-SNE-2"
    df["emb-x"], df["emb-y"] = np.array(lat_emb)[:, 0], np.array(lat_emb)[:, 1]

    selection = altair.selection_multi(fields=["id"])
    color = altair.condition(
        selection, altair.Color("id:N"), altair.value("lightgray")
    )
    chart = (
        altair.Chart(df)
        .mark_point(size=50, opacity=0.4)
        .encode(
            altair.X("emb-x", title=titlex),
            altair.Y("emb-y", title=titley),
            # altair.Color('id', title='class'),
            # altair.Opacity('mode', type='nominal',
            #               scale=altair.Scale(domain=['train', 'val', 'test'],
            #               range=['0.2', '0.4', '1.0']),
            #               title='dataset'),
            altair.Shape(
                "mode",
                scale=altair.Scale(range=["square", "circle", "triangle"]),
                title="mode",
            ),
            altair.Tooltip(
                ["id", "meta", "mode", "avg", "theta:N", "image"]
            ),  # *degrees_of_freedom, 'image']),
            color=color,
        )
        .interactive()
        .properties(width=800, height=500)
        .add_selection(selection)
    )

    if not os.path.exists("plots/latent_embeds"):
        os.mkdir("plots/latent_embeds")
    if embedding == "umap":
        chart.save(
            f"plots/latent_embeds/plt_latent_embed_epoch_{epoch}_umap.html"
        )
    elif embedding == "tsne":
        chart.save(
            f"plots/latent_embeds/plt_latent_embed_epoch_{epoch}_tsne.html"
        )


def accuracy_plot(y_train, ypred_train, y_val, ypred_val):
    print("\n################################################################")
    print("Visualising confusion ...\n")

    ConfusionMatrixDisplay.from_predictions(y_train, ypred_train)
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.savefig("plots/confusion_train.png")

    ConfusionMatrixDisplay.from_predictions(y_val, ypred_val)
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.savefig("plots/confusion_valid.png")


def loss_plot(epochs, train_loss, val_loss=None, p=None):
    #print("\n################################################################")
    #print("Visualising loss ...\n")

    train_loss = np.transpose(np.asarray(train_loss))
    if val_loss is not None:
        val_loss = np.transpose(np.asarray(val_loss))

    cols = ["blue", "red", "green", "orange"]
    labs = [
        "Total loss",
        "Reconstruction loss",
        "KL divergence loss x BETA",
        "Affinity loss x GAMMA",
    ]
    vlabs = [
        "VAL Total loss",
        "VAL Reconstruction loss",
        "VAL KL divergence loss x BETA",
        "VAL Affinity loss x GAMMA",
    ]

    plt.clf()
    plt.ticklabel_format(useOffset=False)
    for i, loss in enumerate(train_loss):
        s = "-"
        plt.plot(
            range(1, epochs + 1), loss, c=cols[i], linestyle=s, label=labs[i]
        )
    if val_loss is not None:
        for i, loss in enumerate(val_loss):
            s = "--"
            plt.plot(
                range(1, epochs + 1),
                loss,
                c=cols[i],
                linestyle=s,
                label=vlabs[i],
            )
    plt.yscale("log")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    if p is not None:
        if len(p) != 7:
            print(
                "WARNING: Function vis.loss_plot is expecting 'p' parameter "
                "to be a list of 7 hyperparameters: batch size, depth, "
                "channel init, latent dimension, learning rate, beta, gamma. "
                "Exiting.\n"
            )
            return
        plt.title(
            "bs: %d, d: %d, ch: %d, lat: %d, lr: %.3f, beta: %.1f, "
            "gamma: %.1f" % (p[0], p[1], p[2], p[3], p[4], p[5], p[6])
        )
    else:
        plt.title("Loss")
    plt.tight_layout()
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.savefig("plots/loss.png")


def recon_plot(img, rec, name="trn"):
    print("\n################################################################")
    print("Visualising reconstructions ...\n")

    fname_in = "plots/" + str(name) + "_recon_in.png"
    fname_out = "plots/" + str(name) + "_recon_out.png"

    img = img[:, :, :, :, img.shape[-1] // 2]
    rec = rec[:, :, :, :, img.shape[-1] // 2]

    plt.subplots(figsize=(10, 10))
    img = torchvision.utils.make_grid(img.cpu(), 10, 2).numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))  # channels last
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.savefig(fname_in)
    plt.close()

    plt.subplots(figsize=(10, 10))
    rec = torchvision.utils.make_grid(rec.detach().cpu(), 10, 2).numpy()
    plt.imshow(np.transpose(rec, (1, 2, 0)))  # channels last
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.savefig(fname_out)
    plt.close()


def latent_disentamglement_plot(lats, vae, device, poses=None):
    print("\n################################################################")
    print("Visualising latent content disentanglement ...\n")

    lats = np.asarray(lats)
    if poses is not None:
        poses = np.asarray(poses)

    lat_means = np.mean(lats, axis=0)
    lat_stds = np.std(lats, axis=0)
    lat_dims = lats.shape[-1]
    lat_grid = np.zeros((lat_dims * 7, lat_dims))
    if poses is not None:
        pos_means = np.mean(poses, axis=0)
        pos_dims = poses.shape[-1]
        pos_grid = np.zeros((lat_dims * 7, pos_dims)) + pos_means

    # Generate vectors representing single transversals along each lat_dim
    for l_dim in range(lat_dims):
        for grid_spot in range(7):
            means = copy.deepcopy(lat_means)
            # every 0.4 interval from -1.2 to 1.2 sigma
            means[l_dim] += lat_stds[l_dim] * (-1.2 + 0.4 * grid_spot)
            lat_grid[l_dim * 7 + grid_spot, :] = means

    # Decode interpolated vectors
    with torch.no_grad():
        lat_grid = torch.FloatTensor(np.array(lat_grid))
        lat_grid = lat_grid.to(device)
        if poses is not None:
            pos_grid = torch.FloatTensor(np.array(pos_grid))
            pos_grid = pos_grid.to(device)
            recon = vae.decoder(lat_grid, pos_grid)
        else:
            recon = vae.decoder(lat_grid, None)
    dsize = recon.shape[-3:]
    if len(dsize) == 0:
        print(
            "WARNING: All images need to be the same size to create "
            "interpolation plot. Exiting.\n"
        )
        return

    recon = np.reshape(np.array(recon.cpu()), (lat_dims, 7, *dsize))

    grid_for_napari = np.zeros(
        (
            recon.shape[2] * recon.shape[0],
            recon.shape[3] * recon.shape[1],
            recon.shape[4],
        ),
        dtype=np.float32,
    )

    # Create and save the mrc file with single transversals
    for i in range(recon.shape[0]):
        for j in range(recon.shape[1]):
            grid_for_napari[
                i * dsize[0] : (i + 1) * dsize[0],
                j * dsize[1] : (j + 1) * dsize[1],
                :,
            ] = recon[i, j, :, :, :]

    if not os.path.exists("plots"):
        os.mkdir("plots")
    with mrcfile.new(
        "plots/disentanglement-latent.mrc", overwrite=True
    ) as mrc:
        mrc.set_data(grid_for_napari)


def pose_disentanglement_plot(lats, poses, vae, device):
    print("\n################################################################")
    print("Visualising pose disentanglement ...\n")
    lats = np.asarray(lats)
    poses = np.asarray(poses)

    pos_means = np.mean(poses, axis=0)
    pos_stds = np.std(poses, axis=0)
    pos_dims = poses.shape[-1]
    pos_grid = np.zeros((pos_dims * 7, pos_dims))

    lat_means = np.mean(lats, axis=0)
    lat_dims = lats.shape[-1]
    lat_grid = np.zeros((pos_dims * 7, lat_dims)) + lat_means

    # Generate vectors representing single transversals along each lat_dim
    for p_dim in range(pos_dims):
        for grid_spot in range(7):
            means = copy.deepcopy(pos_means)
            means[p_dim] += pos_stds[p_dim] * (-1.2 + 0.4 * grid_spot)
            pos_grid[p_dim * 7 + grid_spot, :] = means

    # Decode interpolated vectors
    with torch.no_grad():
        lat_grid = torch.FloatTensor(np.array(lat_grid))
        lat_grid = lat_grid.to(device)
        pos_grid = torch.FloatTensor(np.array(pos_grid))
        pos_grid = pos_grid.to(device)
        recon = vae.decoder(lat_grid, pos_grid)
    dsize = recon.shape[-3:]
    if len(dsize) == 0:
        print(
            "WARNING: All images need to be the same size to create "
            "interpolation plot. Exiting.\n"
        )
        return

    recon = np.reshape(
        np.array(recon.cpu()),
        (pos_dims, 7, *dsize),
    )

    grid_for_napari = np.zeros(
        (
            recon.shape[2] * recon.shape[0],
            recon.shape[3] * recon.shape[1],
            recon.shape[4],
        ),
        dtype=np.float32,
    )

    # Create and save the mrc file with single transversals
    for i in range(recon.shape[0]):
        for j in range(recon.shape[1]):
            grid_for_napari[
                i * dsize[0] : (i + 1) * dsize[0],
                j * dsize[1] : (j + 1) * dsize[1],
                :,
            ] = recon[i, j, :, :, :]

    if not os.path.exists("plots"):
        os.mkdir("plots")
    with mrcfile.new("plots/disentanglement-pose.mrc", overwrite=True) as mrc:
        mrc.set_data(grid_for_napari)


def interpolations_plot(lats, classes, vae, device, poses=None):
    print("\n################################################################")
    print("Visualising interpolations ...\n")
    lats = np.asarray(lats)
    classes = np.asarray(classes)
    if poses is not None:
        poses = np.asarray(poses)

    class_ids = np.unique(classes)
    if len(class_ids) <= 3:
        print(
            "WARNING: Interpolation plot needs at least 4 distinct classes, "
            "cannot visualise interpolations. Exiting.\n"
        )
        return

    class_reps_lats = np.take(
        lats, [np.where(classes == i)[0][0] for i in class_ids], axis=0
    )
    class_reps_lats = np.asarray(class_reps_lats)
    if poses is not None:
        class_reps_poses = np.asarray(
            np.take(
                poses,
                [np.where(classes == i)[0][0] for i in class_ids],
                axis=0,
            )
        )

    draw_four = random.sample(list(enumerate(class_reps_lats)), k=4)
    inds, class_rep_lats = list(zip(*draw_four))
    class_rep_lats = np.asarray(class_rep_lats)
    latent_dim = class_rep_lats.shape[1]
    if poses is not None:
        class_rep_poses = np.asarray([class_reps_poses[i] for i in inds])
        poses_dim = class_rep_poses.shape[1]

    # Generate a gird of latent vectors interpolated between reps of four ids
    grid_size = 6
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x, y)
    corners = [[0, 0], [0, 1], [1, 0], [1, 1]]
    layers_l = []
    for i, corner in enumerate(corners):
        a, b = corner
        layers_l.append(
            np.where(
                1 - (np.sqrt((xx - a) ** 2 + (yy - b) ** 2)) > 0,
                1 - (np.sqrt((xx - a) ** 2 + (yy - b) ** 2)),
                0,
            )[np.newaxis, :, :]
        )
    layers = np.concatenate(
        [layers_l[0], layers_l[2], layers_l[1], layers_l[3]], axis=0
    )
    layers = (layers / np.sum(layers, axis=0))[:, :, :, np.newaxis]
    lat_vecs = class_rep_lats[:, np.newaxis, np.newaxis, :]
    if poses is not None:
        lat_pose = class_rep_poses[:, np.newaxis, np.newaxis, :]

    latents = np.reshape(np.sum((layers * lat_vecs), axis=0), (-1, latent_dim))
    if poses is not None:
        poses = np.reshape(
            np.sum((layers * lat_pose), axis=0), (-1, poses_dim)
        )

    # Decode interpolated vectors
    with torch.no_grad():
        latents = torch.FloatTensor(np.array(latents))
        latents = latents.to(device)
        if poses is not None:
            poses = torch.FloatTensor(np.array(poses))
            poses = poses.to(device)
            recon = vae.decoder(latents, poses)
        else:
            recon = vae.decoder(latents, None)
    dsize = recon.shape[-3:]
    if len(dsize) == 0:
        print(
            "WARNING: All images need to be the same size to create "
            "interpolation plot. Exiting.\n"
        )
        return

    recon = np.reshape(np.array(recon.cpu()), (grid_size, grid_size, *dsize))

    grid_for_napari = np.zeros(
        (
            recon.shape[2] * recon.shape[0],
            recon.shape[3] * recon.shape[1],
            recon.shape[4],
        ),
        dtype=np.float32,
    )

    # Create an mrc file with interpolations
    for i in range(recon.shape[0]):
        for j in range(recon.shape[1]):
            grid_for_napari[
                i * dsize[0] : (i + 1) * dsize[1],
                j * dsize[0] : (j + 1) * dsize[1],
                :,
            ] = recon[i, j, :, :, :]

    if not os.path.exists("plots"):
        os.mkdir("plots")
    with mrcfile.new("plots/interpolations.mrc", overwrite=True) as mrc:
        mrc.set_data(grid_for_napari)


def plot_affinity_matrix(lookup, all_classes, selected_classes):
    """
    This function plots the Affinity matrix and highlights the
    classes selected for the given calculation.

    Parameters
    ----------
    all_classes : All existing classes in the affinity matrix  affinity*.csv
    lookup :  The affinity matrix
    selected_classes : All classes selected by the user for training in classes.csv
    """
    print("\n################################################################")
    print("Visualising Affinity_Matrix ...\n")

    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(lookup, vmin=-1, vmax=1, cmap=plt.cm.get_cmap("RdBu"))
    ax.set_title("Affinity Matrix")
    ax.set_xticks(np.arange(0, len(all_classes)))
    ax.set_xticklabels(all_classes)
    ax.set_yticks(np.arange(0, len(all_classes)))
    ax.set_yticklabels(all_classes)

    # Colour the label for selected classes
    for i, c in enumerate(all_classes):
        if c in selected_classes:
            ax.get_xticklabels()[i].set_color("red")
            ax.get_yticklabels()[i].set_color("red")

    ax.tick_params(axis="x", rotation=90)
    fig.colorbar(im, ax=ax)
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.savefig("plots/Affinity_Matrix.png", dpi=144)
    print("################################################################\n")


def plot_classes_distribution(data, category):
    """Plot histogram with classes distribution"""

    print("\n################################################################")
    print("Visualising Classes Distribution ...\n")

    fig, ax = plt.subplots(figsize=(9, 9))
    labels, counts = np.unique(data, return_counts=True)
    ticks = range(len(counts))
    plt.bar(ticks, counts, align="center", color="blue", alpha=0.5)
    plt.xticks(ticks, labels)
    plt.title("Classes Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Entries")
    plt.savefig("plots/Classes_Distribution_" + category + ".png", dpi=144)
    print("################################################################\n")
