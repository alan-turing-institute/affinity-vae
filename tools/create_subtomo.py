import mrcfile
import warnings
import pandas as pd
from scipy.ndimage import rotate
import torch
import numpy as np
import os 


def create_subtomo(input_path, output_path, datatype,  annot_path , vox_size= [32,32,32],bandpass=False, low_freq = 0  , high_freq = 15,gaussian_blur= False,
                  add_noise =False , noise_int = 0, padding=False, augment, aug_th_min
    ):

    """Function to train an AffinityVAE model. The inputs are training configuration parameters. In this function the
    data is loaded, selected and split into training, validation and test sets, the model is initialised and trained
    over epochs, the results are evaluated visualised and saved and the epoch level with a frequency configured with
    input parameters.

    Parameters
    ----------
    input_path: str
        Path to the folder containing the full tomogram/image.
    output_path: str
        Path to the folder for output subtomograms 
    datatype : str
        data file formats : mrc, npy
    annot_path : str
        path to the folder containing the name of the particles and their x,y,z coordinates 
        file names should be identical to that of the full tomograms with the extension of txt 
        for mat of the txt files is ,  'class','z','y','x','cx','cy','cz
    vox_size: list 
        size of each subtomogram voxel given as a list where vox_size: [x,y,x]
    bandpass: bool 
        Apply band pass filter to the full tomogram before extracting the subtomograms 
    low_freq: int
        Lower frequency threshold for the band pass filter 
    high_freq: int
        higher frequency threshold for the band pass filter 

    """

    # the name of all full tomograms 
    file_list = [
        f for f in os.listdir(input_path) if "." + datatype in f
    ]

    # the name of all annotations for tomograms 
    annot_list = [f for f in os.listdir(input_path) if "." + "txt" in f
    ]

    if len(annot_list) != len(file_list):
        raise RuntimeError("number of full tomograms do not match the number of annotation files"
        )


    for t, tomo  in enumerate(file_list):
        data = read_mrc(os.path.join(input_path, tomo))
        if bandpass:
            bandpass_filter_image = bandpass_filter(data.shape, low_freq,high_freq)
            data = apply_bandpass_filter(data, bandpass_filter_image)
        if add_noise:
            data = add_noise(data, noise_int)
        if gaussian_blur:
            print("gaussian blur is not implemented yet")
            data =data
        

     




def bandpass_filter(image_size,bp_low,bp_high):

    bandpass_filter = np.zeros((image_size), dtype=np.float32)
    if len(image_size) == 2:
        for u in range(image_size[0]):
            for v in range(image_size[1]):
                D = np.sqrt(
                    (u - image_size[0] / 2) ** 2
                    + (v - image_size[1] / 2) ** 2
                )
                if D <= bp_low:
                    bandpass_filter[u, v] = 1
                elif D >= bp_high:
                    bandpass_filter[u, v] = 1
    elif len(image_size) == 3:
        for u in range(image_size[0]):
            for v in range(image_size[1]):
                for w in range(image_size[2]):
                    D = np.sqrt(
                        (u - image_size[0] / 2) ** 2
                        + (v - image_size[1] / 2) ** 2
                        + (w - image_size[2] / 2) ** 2
                    )

                    if D <= bp_low:
                        bandpass_filter[u, v, w] = 1
                    elif D >= bp_high:
                        bandpass_filter[u, v, w] = 1
    return bandpass_filter


def apply_bandpass_filter(image,bandpass_filter):
    F = np.fft.fftn(image)
    Fshift = np.fft.fftshift(F) 
    Gshift = Fshift * bandpass_filter
    G = np.fft.ifftshift(Gshift)
    filtered_image = abs(np.fft.ifftn(G))
    return filtered_image.astype('float32')


def add_noise(input_a,method== 'gaussian',scale):

    if method == 'gaussian':
        # Gaussian distribution parameters
        mean = 0
        var = 0.1
        sigma = var ** 0.5

        gaussian = np.random.normal(mean, sigma, (input_a.shape[0],input_a.shape[1], input_a.shape[1])) 

        output_a = input_a + gaussian * scale
    return output_a.astype(np.float32)

def read_mrc(path):
    '''
    Takes a path and read a mrc file convert the data to np array 
    '''
    warnings.simplefilter('ignore') # to mute some warnings produced when opening the tomos
    with mrcfile.open(path, mode='r+', permissive=True) as mrc:
        mrc.update_header_from_data()

        mrc.header.map = mrcfile.constants.MAP_ID 
        mrc = mrc.data       

    with mrcfile.open(path) as mrc:
        data = np.array(mrc.data)
    return data



def particles_GT(annot,output_path):

    labels = np.loadtxt(annot,dtype='str')

    labels = np.reshape(labels,(-1,7))
    df = pd.DataFrame (labels)
    df.columns =['class','z','y','x','cx','cy','cz']

    df = df.astype({'x':'int'})
    df = df.astype({'y':'int'})
    df = df.astype({'z':'int'})
    proteins = df['class'].unique()
    
    df.drop(df[df['class'] == 'vesicle'].index, inplace = True)

    with open(f"{folder_to_copy_to}/classes.csv", 'w') as f:
        f.write(','.join(proteins))
    return df

def delete_detection_in_edges(df, X_dim,Y_dim,Z_dim): 
    df.drop(df[df['x'] > X_dim-slice_size].index, inplace = True)
    df.drop(df[df['y'] > Y_dim-slice_size].index, inplace = True)
    df.drop(df[df['z'] > Z_dim-slice_size].index, inplace = True)
    df.drop(df[df['x'] <   slice_size].index, inplace = True)
    df.drop(df[df['y'] <   slice_size].index, inplace = True)
    df.drop(df[df['z'] <   slice_size].index, inplace = True)
    return df
