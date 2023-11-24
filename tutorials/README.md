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
python mnist_saver.py mnist.pkl.gz .
```

here the first argument is the path to the downloaded MNIST dataset, and the
second argument is the path to the directory where the processed dataset will be
saved. In this example, we are saving the dataset in the current directory under
`mnist_data`.

Now we are ready to run AffinityVAE on the MNIST dataset. For this we need the
data divided into training and test sets, we need a configuration file
(mnist_config.yml), and we need the classes_mnist.csv and affinity_mnist.csv
files, which are provided in his tutorial. All these files can be found in the
`mnist_data` directory.

To train AffinityVAE on our rotated MNIST dataset, run the following command:

```bash
python path/to/affinity-vae/run.py --config_file mnist_data/mnist_config.yml ---new_out
```

This will train AffinityVAE on the MNIST dataset and save the results in a new
directory created by the `--new_out` flag. In this case the run is configure by
the `mnist_config.yml` file, however, you can also configure the run by passing
arguments to the `run.py` script as shown in the main README file and in the
following example:

```bash
python path/to/affinity-vae/run.py --config_file mnist_data/mnist_config.yml --beta 0.1 --gamma 0.01 --lr 0.001 --epochs 200 --new_out
```

Once the training finishes you can evaluate the model on the test set by
stepping into the new directory and running

```bash
cd path/to/new_out
python path/to/affinity-vae/run.py --config_file ../mnist_data/mnist_config.yml  --data_path ../mnist_data/images_test/ --eval
```

You can also restart training from a checkpoint by running

```bash
cd path/to/new_out
python path/to/affinity-vae/run.py --config_file ../mnist_data/mnist_config.yml  --restart --epochs 2000 --data_path ../mnist_data/images_train/
```

here epochs are set to 2000 to continue training for 1000 extra epochs (assuming
initial training ran for the 1000 epochs defined in the config file).

# Outptus of the training run

The training run will create a directory with the following structure:

```
new_out
├── configs
│   └── timestamp_config.yml
├── logs
│   └── timestamp.log
├── plots
│  
        └──many_plots_produced_during_training.png
│  
├── latents
│   └── epoch_level_html_latents.html
├── states
│   └── epoch_level_model_states.pkl


```
