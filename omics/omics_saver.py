import os
# import random

import numpy as np
import pandas as pd

# input variables - these should be flags - input arguments when you run the script

mat_path = '/Users/ep/Documents/1_datasets/dataset_ibd/output/ol_removed_filtered/vst.counts.fil.ncbi.csv'
labels_path = '/Users/ep/Documents/1_datasets/dataset_ibd/output/ol_removed/ibd.class.vector.csv'
output_path = '/Users/ep/Documents/1_datasets/aff_vae/affinity-vae-omics/omics/omics_data/'

# read in counts matrix and classes

data_matrix = pd.read_csv(mat_path, header=0)
sample_classes = pd.read_csv(labels_path, header=0).values.flatten()

# save the unique list of classes as a csv file

np.savetxt(output_path + 'class_lst.csv', np.unique(sample_classes), delimiter=",")


# function to split counts matrix into a set of 1D np arrays, append class label and save as individual files


def matrixsplitsave(omic_mat, class_vector):
    sample_names_vec = omic_mat.columns

    gene_array_lst = list(range(omic_mat.shape[1]))

    for idx in range(omic_mat.shape[1]):
        gene_array_lst[idx] = omic_mat.iloc[:, idx].values

    # save each array with a file name prefixed with its class label

    for sid, sample, label in zip(sample_names_vec, gene_array_lst, class_vector):
        sample_name = str(label) + '_' + str(sid) + '.npy'
        sample = np.array(sample)

        # Build save path
        save_path = os.path.join(output_path, 'input_arrays', sample_name)

        # save the sample arrays
        np.save(save_path, sample)


matrixsplitsave(data_matrix, sample_classes)


# parse arguments to extract file paths for saving down the data

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    # -db DATABSE -u USERNAME -p PASSWORD -size 20
    parser.add_argument("--matrix_file", help="Path to rna-seq counts matrix csv file")
    parser.add_argument("--labels_file", help="Path to class labels csv file")
    parser.add_argument("--output_path", help="Output path to save data")

    args = parser.parse_args()

    matrix_path = args.matrix_file
    labels_path = args.labels_file
    output_path = args.output_path

    # data = load_mnist(mnist_path)

# randomised train test split
# gene_names = omic_mat.index
# save the vector of the class labels as a csv file
