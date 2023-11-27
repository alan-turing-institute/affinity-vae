# AffinityVAE Tutorial

## Running AffinityVAE on the MNIST dataset

### 1. Install AffinityVAE

As described in the [README](../README.md), AffinityVAE can be installed using
pip:

```bash
python -m venv env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

### 2. Download the MNIST dataset

The MNIST dataset can be downloaded from:
https://figshare.com/articles/dataset/mnist_pkl_gz/13303457

In this tutorial, we assume that the dataset is downloaded to this `tutorials`
directory.

### Create the MNSIT dataset for affinityVAE

We are MNIST dataset with augmentation and rotations, and saved with a structure
that affinityVAE can read. This can be done by running the following command:

```bash
 python mnist_saver.py --mnist_file mnist.pkl.gz --output_path .
```

here the first argument is the path to the downloaded MNIST dataset, and the
second argument is the path to the directory where the processed dataset will be
saved. In this example, we are saving the dataset in the current directory under
`mnist_data`.

Now we are ready to run AffinityVAE on the MNIST dataset. For this we need the
data divided into training and test sets, we need a configuration file
(`mnist_config.yml`), and we need the `classes_mnist.csv` and
`affinity_mnist.csv` files, which are provided in his tutorial. All these files
can be found in the `mnist_data` directory.

First thing you need to do is to modify the `mnist_config.yml` file to point to
the absolute paths of the data, classes, and affinity files. Currently, the
paths in the config file are relative this directory but this can cause issues
when running the code from a different directory. To modify the paths, open the
`mnist_config.yml` file and change the following lines:

```yaml
datapath: /absolute/path/to/mnist_data/images_train/
classes: /absolute/path/to/mnist_data/classes_mnist.csv
affinity: /absolute/path/to/mnist_data/affinity_mnist.csv
```

in general, we recommend to work with absolute paths to avoid issues.

To train AffinityVAE on our rotated MNIST dataset, run the following command:

```bash
python path/to/affinity-vae/run.py --config_file /absolute/path/to/mnist_data/mnist_config.yml ---new_out
```

This will train AffinityVAE on the MNIST dataset and save the results in a new
directory created by the `--new_out` flag. In this case the run is configured by
the `mnist_config.yml` file, however, you can also configure the run by passing
arguments to the `run.py` script as shown in the main README file and in the
following example:

```bash
python path/to/affinity-vae/run.py --config_file /absolute/path/to/mnist_data/mnist_config.yml --beta 0.1 --gamma 0.01 --lr 0.001 --epochs 200 --new_out
```

Here the command line arguments override the values in the config file.

Once the training finishes you can evaluate the model on the test set by
stepping into the new directory and running

```bash
cd path/to/new_out
python path/to/affinity-vae/run.py --config_file /absolute/path/to/mnist_data/mnist_config.yml  --data_path /absolute/path/to/mnist_data/images_test/ --eval
```

You can also restart training from a checkpoint by running

```bash
cd path/to/new_out
python path/to/affinity-vae/run.py --config_file /absolute/path/to/mnist_data/mnist_config.yml  --restart --epochs 2000 --data_path /absolute/path/to/mnist_data/images_train/
```

here epochs are set to 2000 to continue training for 1000 extra epochs (assuming
initial training ran for the 1000 epochs defined in the config file).

# Outptus of the training run

The training run will create a directory with the following structure:

```
new_out
├── configs # copy of the config file used for the run for reproducibility
├── logs # run logs
├── plots # plots and data of the training and evaluation metrics
├── latents # html files latent space of the training and test sets, these files can be very large, so we recomend them to only runnin the at evaluation time (using the --dynamic flag)
├── states #saving checkpoints of the models and the training latent space to be use for evaluation or restart training


```
