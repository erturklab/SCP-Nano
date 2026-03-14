import os 
import cv2 
import numpy as np
import re
import pickle
import nibabel as nib
from pathlib import Path

def psave(path, variable):
    '''
    psave(path, variable)
    
    Takes a variable (given as string with its name) and saves it to a file as specified in the path.
    The path must at least contain the filename (no file ending needed), and can also include a 
    relative or an absolute folderpath, if the file is not to be saved to the current working directory.
    '''
    path = Path(path)

    if path.suffix != ".pickledump":
        path = path.with_suffix(".pickledump")

    # create folder if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(variable, f, protocol=4)


def pload(path):
    '''
    variable = pload(path)
    
    Loads a variable from a file that was specified in the path. The path must at least contain the 
    filename (no file ending needed), and can also include a relative or an absolute folderpath, if 
    the file is not to located in the current working directory.
    '''
    path = Path(path)
    if path.suffix != ".pickledump":
        path = path.with_suffix(".pickledump")

    path = path.resolve()

    with open(path, "rb") as f:
        return pickle.load(f)

def estimateVolSize(pathtofolders,folderlist):
    ''' 
    Estimates size of stack of images from arbitrary number of folders where each folder 
    corresponds to one channel. Expects each file per folder to be grayscale
     
    Returns string with size in GB, dimensionality (n_y,n_x,n_z,n_c), and bitdepth, where
    
     * n_y is the height of a single image
     * n_x is the width of a single image
     * n_z is the number of images per folder
     * n_c is the number of folders
    '''

    print("Path for volume size estimation: {} {}".format(pathtofolders, folderlist))
    pathtofolders = Path(pathtofolders)
    n_c = len(folderlist)
    if n_c > 1:
        folder = pathtofolders / folderlist[0]
    else:
        folder = pathtofolders
    filelist = sorted([f for f in folder.iterdir() if f.is_file()])
    if not filelist:
        raise ValueError(f"No files found in {folder}")
        
    n_z = len(filelist)
    path = filelist[0]
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    n_y, n_x = image.shape
    bitdepth = image.dtype.itemsize * 8

    size_bytes = n_y * n_x * n_z * n_c * bitdepth / 8
    size_gb = size_bytes / (1024 ** 3)
    
    return (size_gb,(n_y,n_x,n_z,n_c),bitdepth)


def writeNifti(path,volume,compress=False):
    '''
    writeNifti(path,volume)
    
    Takes a Numpy volume, converts it to the Nifti1 file format, and saves it to file under
    the specified path. 
    '''
    path = Path(path)

    # Ensure correct extension
    if compress:
        if path.suffix != ".gz":
            path = path.with_suffix(".nii.gz")
    else:
        if path.suffix != ".nii":
            path = path.with_suffix(".nii")

    # Create output folder if necessary
    path.parent.mkdir(parents=True, exist_ok=True)

    # Adjust orientation
    # (y, x, z) -> (x, y, z)
    vol = np.swapaxes(volume, 0, 1)
    # RAI orientation affine
    affine = np.eye(4)
    affine[0, 0] = -1
    affine[1, 1] = -1
    nifti_obj = nib.Nifti1Image(vol, affine)

    nib.save(nifti_obj, str(path))


def readNifti(path,reorient=None):
    '''
    volume = readNifti(path)
    
    Reads in the NiftiObject saved under path and returns a Numpy volume.
    This function can also read in .img files (ANALYZE format).
    '''
    path = Path(path)

    # Try resolving missing extensions
    if not path.exists():
        for ext in (".nii", ".nii.gz"):
            candidate = path.with_suffix(ext)
            if candidate.exists():
                path = candidate
                break
        else:
            raise FileNotFoundError(f"No file found for {path}")

    nifti_obj = nib.load(str(path))
    # Load volume and adjust orientation from (x,y,z) -> (y,x,z)
    volume = np.swapaxes(np.asarray(nifti_obj.dataobj), 0, 1)
    return volume
