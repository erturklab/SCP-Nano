import cv2
import os
import numpy as np
import glob
from utils import filehandling

def compute_organ_intensity_range(path_organ_raw, path_organ_mask, path_output_organ_raw_nifti, organ_key):
    """
    Compute intensity range of a cropped organ and save the masked organ image as nifti file

    path_organ_raw: path of organ raw image sequence
    path_mask: path of organ mask image sequence
    path_output_organ_raw_nifti: path to save masked organ image as nifti file
    organ_key: label value of the organ 
    """
    organ_mask_slicelist = sorted(os.listdir(path_organ_mask))
    intensity_min = 65535
    intensity_max= 0
    z_dim = len(os.listdir(path_organ_raw))
    x_dim, y_dim = cv2.imread(path_organ_raw+os.listdir(path_organ_raw)[0], -1).shape
    organ_mask_raw_arr = np.zeros((x_dim, y_dim, z_dim))
    for z, z_slice in enumerate(sorted(os.listdir(path_organ_raw))):
        img_organ_raw = cv2.imread(path_organ_raw + z_slice, -1)
        img_organ_raw= np.squeeze(img_organ_raw)

        img_organ_mask = cv2.imread(path_organ_mask + organ_mask_slicelist[z], -1)
        img_organ_mask = img_organ_mask==int(organ_key)
        
        organ_mask_raw_slice = img_organ_raw*img_organ_mask
        organ_mask_raw_arr[:, :, z]= organ_mask_raw_slice
        if len(organ_mask_raw_slice[img_organ_mask==True])==0:
            continue          
        if np.min(organ_mask_raw_slice[img_organ_mask==True])<intensity_min:
            intensity_min = np.min(organ_mask_raw_slice[img_organ_mask==True])
        if np.max(organ_mask_raw_slice)>intensity_max:
            intensity_max = np.max(organ_mask_raw_slice)

    print("Intensity range:", intensity_min, intensity_max)
    print("Saving organ raw image nifti")
    filehandling.writeNifti(path_output_organ_raw_nifti, organ_mask_raw_arr)
    return intensity_min, intensity_max

    
# Root of whole-body image data    
dir_wholebody_data = ""
# name of organ to crop into patches 
cur_organ_name = ""


# compute the intensity range of an organ raw image
path_organ_keys           = os.path.join(dir_wholebody_data, "organ_keys.txt")
keys_dict = {}
f_organ_keys = open(path_organ_keys)
line = f_organ_keys.readline()
while line:
    organ_line = line.replace('\n', '')
    organ_key = organ_line.split(":")[0]
    organ_name = organ_line.split(":")[1]
    keys_dict[organ_key] = organ_name
    line = f_organ_keys.readline()
organ_keys_list = list(keys_dict.keys())
organ_names_list = list(keys_dict.values())
cur_organ_key = organ_keys_list[organ_names_list.index(cur_organ_name)]
path_organ_raw = os.path.join(dir_wholebody_data, "organ_results", f"organ_{cur_organ_name}_raw", "")
path_organ_mask = os.path.join(dir_wholebody_data, "organ_results", f"organ_{cur_organ_name}_mask", "")
path_organ_raw_nifti = os.path.join(dir_wholebody_data, "organ_results", f"organ_{cur_organ_name}_raw.nii.gz")
intensity_min, intensity_max = compute_organ_intensity_range(path_organ_raw=path_organ_raw, path_organ_mask=path_organ_mask, 
                                                             path_output_organ_raw_nifti= path_organ_raw_nifti,
                                                             organ_key = cur_organ_key)


# patch normalization based on organ intensity range
path_patches = os.path.join(dir_wholebody_data, "organ_results", f"organ_{cur_organ_name}_crop", "local_C01", "")
path_patches_norm = os.path.join(dir_wholebody_data, "organ_results", f"organ_{cur_organ_name}_crop", "local_C01_norm", "")

if not os.path.exists(path_patches_norm):
  os.mkdir(path_patches_norm) 
  
volumes_list = sorted(glob.glob(os.path.join(path_patches, '*.nii')))
print(len(volumes_list))
for myfile in volumes_list:    
    volume= filehandling.readNifti(myfile)
    volume=(volume-intensity_min)/(intensity_max-intensity_min)
    filehandling.writeNifti(myfile.replace(path_patches, path_patches_norm).replace('.nii', '_0000.nii.gz'), volume)


