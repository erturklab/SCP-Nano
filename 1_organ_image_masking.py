import cv2
import os
import numpy as np
import pickle
import pandas as pd
import imageio as io

def _get_bb(img:np.array, index:int):
    """
    Get the bounding box of a numpy array for a given index

    img: 2d or 3d numpy array of organ mask
    index: select one organ by its label value in the mask to calculate its bounding box 
    """

    if len(img.shape) == 2:
        x = np.any(img == index, axis=1)
        y = np.any(img == index, axis=0)

        xmin, xmax = np.where(x)[0][[0, -1]]
        ymin, ymax = np.where(y)[0][[0, -1]]

        return xmin, xmax, ymin, ymax
    else:
        x = np.any(img == index, axis=(1, 2))
        y = np.any(img == index, axis=(0, 2))
        z = np.any(img == index, axis=(0, 1))

        xmin, xmax = np.where(x)[0][[0, -1]]
        ymin, ymax = np.where(y)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]

        return xmin, xmax, ymin, ymax, zmin, zmax

def get_organ_name(key):
    global keys
    return keys[str(key)]

def get_organ_bb(path_organ:str,path_output:str):
    """
    Get the bounding boxes for individual organs 
    
    path_organ: path of organ mask tiff sequence
    path_output: path to save the bounding box info of every organ in a pickle file
    """
    bb_indices = {}
    organ_slicelist = sorted(os.listdir(path_organ))
    for z, z_slice in enumerate(organ_slicelist):
        # Load images
        img_organ = cv2.imread(path_organ + organ_slicelist[z], -1)
        img_organ = np.squeeze(img_organ)

        # Get uniques for each z_slice
        uniques = np.unique(img_organ)
        organ_slice_name = organ_slicelist[z]
        print(f"{organ_slice_name} {uniques}")
        uniques = uniques[uniques > 0]

        # Get mask bounding boxes for each unique element
        for mask in uniques:
            xmin, xmax, ymin, ymax = _get_bb(img_organ, index=mask)
            zmin = z
            zmax = z

            # If there are already BBs, update them
            if mask in bb_indices.keys():
                _xmin, _xmax, _ymin, _ymax, _zmin, _zmax = bb_indices[mask]
                if _xmin < xmin:
                    xmin = _xmin
                if _xmax > xmax:
                    xmax = _xmax
                if _ymin < ymin:
                    ymin = _ymin
                if _ymax > ymax:
                    ymax = _ymax
                if _zmin < zmin:
                    zmin = _zmin
                if _zmax > zmax:
                    zmax = _zmax
            bb_indices[mask] = [xmin, xmax, ymin, ymax, zmin, zmax]
        print(f"{z_slice} {bb_indices} {uniques}")

    with open(path_output + "bbs.pickledump", "wb") as file:
        pickle.dump(bb_indices, file, protocol=pickle.HIGHEST_PROTOCOL)

def get_organ_params(path_organ_mask, organ_bbs, path_output):
    """
    Compute the size of every organ (number of pixels)

    path_organ_mask: path of whole-body organ mask sequence
    organ_bbs: pickle file storing bounding box info of every organ 
    path_output: path to save the CSV file that contains the size info of every organ
    """
    with open(organ_bbs, "rb") as file:
        organ_bbs = pickle.load(file)
    print(organ_bbs.keys())

    organs_voxels = {}
    for organ_index in organ_bbs.keys():
        organs_voxels.update({get_organ_name(organ_index): 0})

    max_z = len(os.listdir(path_organ_mask))
    for z, z_slice in enumerate(sorted(os.listdir(path_organ_mask))):
        img_mask = cv2.imread(path_organ_mask + z_slice, -1)
        img_mask = np.squeeze(img_mask)
        for organ_index in organ_bbs.keys():
            organ_name = get_organ_name(organ_index)
            print(f"Selecting Z\t{z}/{max_z}\tOrgan \t{organ_name}", end="\r", flush=True)
            xmin, xmax, ymin, ymax, zmin, zmax = organ_bbs[organ_index]
            try:
                if zmin <= z < zmax:
                        cur_mask = np.zeros_like(img_mask)
                        cur_mask[img_mask==organ_index] = 1
                        organs_voxels[organ_name] = organs_voxels[organ_name] + np.sum(cur_mask)
            except IndexError as ie:
                if zmin <= z < zmax:
                    cur_mask = np.zeros_like(img_mask)
                    cur_mask[img_mask == organ_index] = 1
                    organs_voxels[organ_name] = organs_voxels[organ_name] + np.sum(cur_mask)

    print(organs_voxels)
    df_cnt = pd.DataFrame(list(organs_voxels.items()),
                   columns=['Organ', 'voxel'])
    df_cnt.to_csv(f"{path_output}organs_voxels.csv")


def mask_organs(organs_bbs, path_wholebody_raw, path_wholebody_organ_mask, path_organ_output):
    """
    Cut out the organs, saving the raw tiff and mask tiff sequences of every organ 

    organs_bbs: pickle file storing bounding box info of every organ 
    path_wholebody_raw: path of whole-body raw image sequence
    path_wholebody_organmask: path of whole-body organ mask sequence
    path_organ_output: path to save the raw tiff and mask tiff sequences of every organ 
    """
    # Load the pickled bb dict
    with open(organs_bbs, "rb") as file:
        organs_bbs = pickle.load(file)

    if not os.path.exists(path_organ_output):
        os.mkdir(path_organ_output)

    print(f"Cropping out raw tiffs and mask tiffs of every organ")
    print(f"Keys {organs_bbs.keys()}")

    for organ_index in organs_bbs.keys():
        organ_name = get_organ_name(organ_index)
        # Create folder to save raw tiffs for every organ
        if not os.path.exists(os.path.join(path_organ_output, 'organ_' + organ_name + '_raw')):
            os.mkdir(os.path.join(path_organ_output, 'organ_' + organ_name + '_raw'))
        # Create folder to save mask tiffs for every organ
        if not os.path.exists(os.path.join(path_organ_output, 'organ_' + organ_name + '_mask')):
            os.mkdir(os.path.join(path_organ_output, 'organ_' + organ_name + '_mask'))

    # Go through z slices, mask them for each organ
    max_z = len(os.listdir(path_wholebody_raw))
    for z, z_slice in enumerate(sorted(os.listdir(path_wholebody_raw))):

        img_raw = cv2.imread(path_wholebody_raw + z_slice, -1)
        img_raw = np.squeeze(img_raw)

        img_organ_mask = cv2.imread(path_wholebody_organ_mask + sorted(os.listdir(path_wholebody_organ_mask))[z], -1)
        img_organ_mask = np.squeeze(img_organ_mask)

        for organ_index in organs_bbs.keys():
            organ_name = get_organ_name(organ_index)
            print(f"Selecting Z\t{z}/{max_z}\tOrgan \t{organ_name}", end="\r",flush=True)
            xmin, xmax, ymin, ymax, zmin, zmax = organs_bbs[organ_index]
            try:
                if zmin <= z < zmax:
                    
                    z_idx = z-zmin
                    io.imwrite(os.path.join(path_organ_output, 'organ_' + organ_name + '_raw', f"slice_{z_idx:04}.tif"), img_raw[xmin:xmax, ymin:ymax])
                    io.imwrite(os.path.join(path_organ_output, 'organ_' + organ_name + '_mask', f"slice_{z_idx:04}.tif"), img_organ_mask[xmin:xmax, ymin:ymax])
            except IndexError as ie:
                if zmin <= z < zmax:
                    z_idx = z-zmin
                    io.imwrite(os.path.join(path_organ_output, 'organ_' + organ_name + '_raw', f"slice_{z_idx:04}.tif"), img_raw[xmin:xmax, ymin:ymax])
                    io.imwrite(os.path.join(path_organ_output, 'organ_' + organ_name + '_mask', f"slice_{z_idx:04}.tif"), img_organ_mask[xmin:xmax, ymin:ymax])

    print("Done.") 


# Root of whole-body image data    
dir_wholebody_data = ""

path_keys           = os.path.join(dir_wholebody_data, "organ_keys.txt")
# Path of the whole-body raw tiff sequence
path_raw            = os.path.join(dir_wholebody_data, "C01", "")

# Path of the organ annotation tiff sequence
path_organ_mask          = os.path.join(dir_wholebody_data, "organmask", "")

# Where to save cropped organ images and masks
path_organ_output         = os.path.join(dir_wholebody_data, "organresults", "")


# Check the length of whole-body raw sequence and organ mask sequence to be the same
assert len(os.listdir(path_raw))==len(os.listdir(path_organ_mask)), 'Whole-body raw sequence and organ mask sequence have different length!'

keys                = {}
f = open(path_keys)
line = f.readline()
while line:
    organ_line = line.replace('\n', '')
    organ_key = organ_line.split(":")[0]
    organ_name = organ_line.split(":")[1]
    keys[organ_key] = organ_name
    line = f.readline()


# Calculate the bounding boxes of organs
get_organ_bb(path_organ=path_organ_mask, path_output=dir_wholebody_data)

# Croped out raw images and mask images of organs 
mask_organs(organs_bbs=os.path.join(dir_wholebody_data, "bbs.pickledump"), 
            path_wholebody_raw=path_raw, path_wholebody_organ_mask=path_organ_mask,path_organ_output=path_organ_output)

# Calculate the size of organs 
get_organ_params(path_organ_mask=path_organ_mask, organ_bbs=os.path.join(dir_wholebody_data, "bbs.pickledump"),
                 path_output=path_organ_output)
 