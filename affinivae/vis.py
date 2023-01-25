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
from sklearn import metrics, preprocessing
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier


def encoder(i):
    """Encode PIL Image as base64 buffer."""
    import base64
    from io import BytesIO

    with BytesIO() as buffer:
        i.thumbnail((110, 110))
        i.save(buffer, "PNG")
        data = base64.encodebytes(buffer.getvalue()).decode("utf-8")

    return f"{data}"


def decoder(i):
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
        raise RuntimeError(
            "Wrong data format, please pass either a single "
            "unsqueezed tensor or a batch to image formatter."
        )
    im *= 255
    im = im.astype(np.uint8)
    if batch:
        # if batch we are adding a reconstruction to an input that already
        # exists in the DF, '&' is a separator.
        return ["&" + encoder(Image.fromarray(i)) for i in im]
    else:
        return encoder(Image.fromarray(im))


def merge(im):
    """Merge 2 base64 buffers as PIL Images and encode back to base64
    buffers."""
    i = im.split("&")
    if len(i) != 2:
        raise RuntimeError(
            "Image format corrupt. Number of images in meta_df: {}".format(
                len(i)
            )
        )

    im1 = decoder(i[0])
    im2 = decoder(i[1])

    new_image = Image.new("L", (im2.size[0] + im1.size[0], im2.size[1]))
    new_image.paste(im1, (0, 0))
    new_image.paste(im2, (im1.size[0], 0))
    data = encoder(new_image)

    return f"data:image/png;base64,{data}"


def vis_latentembed_plot(df, epoch, embedding="umap"):
    print(
        "\n###################################################################"
    )
    print("Visualising embedding {}...\n".format(embedding))

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

    plt.clf()
    plt.scatter(
        df["emb-x"],
        df["emb-y"],
        c=[ord(i[0]) for i in df["id"]],
        label=df["id"],
    )
    plt.savefig("plots/embedding_{}.png".format(str(embedding)))

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
                ["id", "meta", "mode", "avg", "theta", "image"]
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


def vis_loss_plot(
    epochs,
    loss,
    recon_loss=None,
    kldiv_loss=None,
    shape_loss=None,
    val_loss=None,
    p=None,
):
    print(
        "\n###################################################################"
    )
    print("Visualising loss ...\n")

    plt.clf()
    plt.ticklabel_format(useOffset=False)
    plt.plot(
        range(1, epochs + 1),
        loss,
        c="blue",
        label="Total loss on training data",
    )
    plt.plot(
        range(1, epochs + 1),
        recon_loss,
        c="red",
        label="Recon. loss on training data",
    )
    plt.plot(
        range(1, epochs + 1),
        kldiv_loss,
        c="green",
        label="KL divergence loss x beta",
    )
    plt.plot(
        range(1, epochs + 1),
        shape_loss,
        c="yellow",
        label="Shape similarity loss x gamma",
    )
    plt.plot(
        range(1, epochs + 1),
        val_loss,
        c="black",
        label="Recon. loss on validation data",
    )
    plt.yscale("log")
    plt.ylabel("Loss per instance")
    plt.xlabel("Epochs")
    plt.legend()
    if p is not None:
        plt.title(
            "Loss with bs: %d, d: %d, ch: %d, lat: %d, lr: %.3f, beta: %.1f, "
            "gamma: %.1f" % (p[0], p[1], p[2], p[3], p[4], p[5], p[6])
        )
    else:
        plt.title("Loss")
    plt.tight_layout()
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.savefig("plots/loss.png")


def vis_recon_plot(img, rec, val=False):
    print(
        "\n###################################################################"
    )
    print("Visualising reconstructions ...\n")

    img = img[:, :, :, :, img.shape[-1] // 2]
    rec = rec[:, :, :, :, img.shape[-1] // 2]

    plt.subplots(figsize=(10, 10))
    img = torchvision.utils.make_grid(img.cpu(), 10, 2).numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))  # channels last

    if val:
        fname_in = "plots/val_recon_in.png"
        fname_out = "plots/val_recon_out.png"
    else:
        fname_in = "plots/trn_recon_in.png"
        fname_out = "plots/trn_recon_out.png"

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


def vis_single_transversals(
    df, vae, device, patch_size, latent_dims, pose_dims, pose
):
    print(
        "\n###################################################################"
    )
    print("Visualising latent content disentanglement ...\n")

    df_latent = df.filter(regex="lat")  # .reset_index(drop=True)
    mean_all_dims = [
        df_latent[lat_dim_col].mean() for lat_dim_col in df_latent.columns
    ]
    std_all_dims = [
        df_latent[lat_dim_col].std() for lat_dim_col in df_latent.columns
    ]
    latents = np.zeros((latent_dims * 7, latent_dims))
    if pose:
        df_poses = df.filter(regex="pos").reset_index(drop=True)
        mean_all_poses = [
            df_poses[lat_dim_col].mean() for lat_dim_col in df_poses.columns
        ]
        poses = np.zeros((latent_dims * 7, pose_dims)) + mean_all_poses

    # Generate vectors representing single transversals along each lat_dim
    for lat_dim in range(latent_dims):
        for grid_spot in range(7):
            new_vector = copy.deepcopy(mean_all_dims)
            # every 0.4 interval from -1.2 to 1.2 sigma
            new_vector[lat_dim] += std_all_dims[lat_dim] * (
                -1.2 + 0.4 * grid_spot
            )
            latents[lat_dim * 7 + grid_spot, :] = new_vector

    # Decode interpolated vectors
    with torch.no_grad():
        latents = torch.FloatTensor(np.array(latents))
        latents = latents.to(device)
        if pose:
            poses = torch.FloatTensor(np.array(poses))
            poses = poses.to(device)
            interp_recon = vae.decoder(latents, poses)
        else:
            interp_recon = vae.decoder(latents, None)

    interp_recon = np.reshape(
        np.array(interp_recon.cpu()),
        (latent_dims, 7, patch_size[0], patch_size[1], patch_size[2]),
    )

    grid_for_napari = np.zeros(
        (
            interp_recon.shape[2] * interp_recon.shape[0],
            interp_recon.shape[3] * interp_recon.shape[1],
            interp_recon.shape[4],
        ),
        dtype=np.float32,
    )

    # Create and save the mrc file with single transversals
    for i in range(interp_recon.shape[0]):
        for j in range(interp_recon.shape[1]):
            grid_for_napari[
                i * patch_size[0] : (i + 1) * patch_size[0],
                j * patch_size[1] : (j + 1) * patch_size[1],
                :,
            ] = interp_recon[i, j, :, :, :]

    if not os.path.exists("plots"):
        os.mkdir("plots")
    with mrcfile.new("plots/transversals.mrc", overwrite=True) as mrc:
        mrc.set_data(grid_for_napari)


def vis_single_transversals_pose(
    df, vae, device, patch_size, latent_dims, pose_dims
):
    print(
        "\n###################################################################"
    )
    print("Visualising pose disentanglement ...\n")
    df_latent = df.filter(regex="lat").reset_index(drop=True)
    df_poses = df.filter(regex="pos").reset_index(drop=True)
    mean_all_poses = [
        df_poses[lat_dim_col].mean() for lat_dim_col in df_poses.columns
    ]
    std_all_poses = [
        df_poses[lat_dim_col].std() for lat_dim_col in df_poses.columns
    ]
    mean_all_dims = [
        df_latent[lat_dim_col].mean() for lat_dim_col in df_latent.columns
    ]
    poses = np.zeros((pose_dims * 7, pose_dims)) + mean_all_poses
    latents = np.zeros((pose_dims * 7, latent_dims)) + mean_all_dims

    # Generate vectors representing single transversals along each lat_dim
    for pos_dim in range(pose_dims):
        for grid_spot in range(7):
            new_vector = copy.deepcopy(mean_all_poses)
            new_vector[pos_dim] += std_all_poses[pos_dim] * (
                -1.2 + 0.4 * grid_spot
            )
            poses[pos_dim * 7 + grid_spot, :] = new_vector

    # Decode interpolated vectors
    with torch.no_grad():
        latents = torch.FloatTensor(np.array(latents))
        latents = latents.to(device)
        poses = torch.FloatTensor(np.array(poses))
        poses = poses.to(device)
        interp_recon = vae.decoder(latents, poses)

    interp_recon = np.reshape(
        np.array(interp_recon.cpu()),
        (pose_dims, 7, patch_size[0], patch_size[1], patch_size[2]),
    )

    grid_for_napari = np.zeros(
        (
            interp_recon.shape[2] * interp_recon.shape[0],
            interp_recon.shape[3] * interp_recon.shape[1],
            interp_recon.shape[4],
        ),
        dtype=np.float32,
    )

    # Create and save the mrc file with single transversals
    for i in range(interp_recon.shape[0]):
        for j in range(interp_recon.shape[1]):
            grid_for_napari[
                i * patch_size[0] : (i + 1) * patch_size[0],
                j * patch_size[1] : (j + 1) * patch_size[1],
                :,
            ] = interp_recon[i, j, :, :, :]

    if not os.path.exists("plots"):
        os.mkdir("plots")
    with mrcfile.new("plots/pose-transversals.mrc", overwrite=True) as mrc:
        mrc.set_data(grid_for_napari)


def vis_interp_grid(df, vae, device, patch_size, pose):
    print(
        "\n###################################################################"
    )
    print("Visualising interpolations ...\n")

    class_rep_ids = df.id.unique()
    if len(class_rep_ids) <= 3:
        print(
            "WARNING: Not enough distinct classes, cannot visualise "
            "interpolations.\n"
        )
        return
    latents_of_class_reps = [
        df[df.id == pid].iloc[0].filter(regex="lat").values.astype(np.float32)
        for pid in class_rep_ids
    ]
    if pose:
        poses_of_class_reps = [
            df[df.id == pid]
            .iloc[0]
            .filter(regex="pos")
            .values.astype(np.float32)
            for pid in class_rep_ids
        ]

    # Generate a gird of latent vectors interpolated between reps of four
    # poke_ids
    grid_size = 6
    inds, class_rep_vecs = list(
        zip(*random.sample(list(enumerate(latents_of_class_reps)), k=4))
    )
    class_rep_vecs = np.array(class_rep_vecs)
    if pose:
        class_rep_pose = np.array([poses_of_class_reps[i] for i in inds])
    latent_dim = class_rep_vecs.shape[1]
    if pose:
        pose_dim = class_rep_pose.shape[1]
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
    lat_vecs = class_rep_vecs[:, np.newaxis, np.newaxis, :]
    if pose:
        lat_pose = class_rep_pose[:, np.newaxis, np.newaxis, :]
    latents = np.reshape(np.sum((layers * lat_vecs), axis=0), (-1, latent_dim))
    if pose:
        poses = np.reshape(np.sum((layers * lat_pose), axis=0), (-1, pose_dim))

    # Decode interpolated vectors
    with torch.no_grad():
        latents = torch.FloatTensor(np.array(latents))
        latents = latents.to(device)
        if pose:
            poses = torch.FloatTensor(np.array(poses))
            poses = poses.to(device)
            interp_recon = vae.decoder(latents, poses)
        else:
            interp_recon = vae.decoder(latents, None)

    interp_recon = np.reshape(
        np.array(interp_recon.cpu()),
        (grid_size, grid_size, patch_size[0], patch_size[1], patch_size[2]),
    )

    grid_for_napari = np.zeros(
        (
            interp_recon.shape[2] * interp_recon.shape[0],
            interp_recon.shape[3] * interp_recon.shape[1],
            interp_recon.shape[4],
        ),
        dtype=np.float32,
    )

    # Create an mrc file with interpolations
    for i in range(interp_recon.shape[0]):
        for j in range(interp_recon.shape[1]):
            grid_for_napari[
                i * patch_size[0] : (i + 1) * patch_size[1],
                j * patch_size[0] : (j + 1) * patch_size[1],
                :,
            ] = interp_recon[i, j, :, :, :]
    if not os.path.exists("plots"):
        os.mkdir("plots")
    with mrcfile.new("plots/interpolations.mrc", overwrite=True) as mrc:
        mrc.set_data(grid_for_napari)


def vis_accuracy(df):
    print(
        "\n###################################################################"
    )
    print("Visualising confusion ...\n\n")
    le = preprocessing.LabelEncoder()

    train_df = df[df["mode"] == "train"]
    val_df = df[df["mode"] == "val"]

    x_train = train_df.filter(regex="lat")
    y_train = le.fit_transform(train_df.id)

    x_val = val_df.filter(regex="lat")
    y_val = le.fit_transform(val_df.id)

    neigh = KNeighborsClassifier(n_neighbors=20)
    neigh.fit(x_train, y_train)
    y_pred_train = neigh.predict(x_train)
    y_pred_val = neigh.predict(x_val)
    train_acc = metrics.accuracy_score(y_train, y_pred_train)
    val_acc = metrics.accuracy_score(y_val, y_pred_val)

    ConfusionMatrixDisplay.from_predictions(y_train, y_pred_train)
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.savefig("plots/confusion_train.png")

    ConfusionMatrixDisplay.from_predictions(y_val, y_pred_val)
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.savefig("plots/confusion_valid.png")

    return train_acc, val_acc
