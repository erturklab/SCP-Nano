import os
import cv2
import pickle
import numpy as np
import nibabel as nib
from libtiff import TIFFimage

def perorgan_den_to_wholebody(path_density_nifti, path_raw, bb, path_whole_body_out):
    """
    Mapping the density map of an organ back to its position in the whole body 

    path_density_nifti: path of the density map nifti file of the organ
    path_raw: path of the whole body raw image, used to get the inmage size of whole body
    bb: bounding box of the organ in the whole body space, e.g. [x_low, x_up, y_low, y_up, z_low, z_up]
    path_whole_body_out: path to save the mapping-back result
    """
    assert path_density_nifti.find('.nii')!=-1, "Please specify a valid organ density file!"
    organ_den_img = nib.load(path_density_nifti).get_fdata()
    organ_den_img = organ_den_img.transpose(1, 0, 2)
    print(organ_den_img.dtype)
    print(np.min(organ_den_img), np.max(organ_den_img))
    organ_h, organ_w, organ_d = organ_den_img.shape

    x_min, x_max, y_min, y_max, z_min, z_max = bb
    if not os.path.exists(path_whole_body_out):
        os.mkdir(path_whole_body_out)
    print(organ_h, organ_w, organ_d)

    organ_raw_slicelist = sorted(os.listdir(path_raw))
    whole_body_h, whole_body_w = cv2.imread(path_raw + organ_raw_slicelist[0], -1).shape
    whole_body_d = len(organ_raw_slicelist)
    print(whole_body_h, whole_body_w, whole_body_d)
    
    for z in range(len(organ_raw_slicelist)):
        cur_organ_den_slice = np.zeros((whole_body_h, whole_body_w), dtype=np.float32)
        if z>=z_min and z<z_max:
            z_in_organ = z-z_min
            cur_organ_den_slice[x_min:x_max, y_min:y_max] = organ_den_img[:,:,z_in_organ]
        tiff = TIFFimage(cur_organ_den_slice, description='')
        tiff.write_file(f"{path_whole_body_out}/slice_{z:04}.tif", compression='lzw', verbose=False)

def combine_multichannel(dir_multichannel_list, path_combined_out):
    """
    Merge the density maps of different organs in one channel 
    
    dir_multichannel_list: path list of density maps (tiff squences) of different organs
    path_combined_out: path of the merge density map
    """
    if not os.path.exists(path_combined_out):
        os.mkdir(path_combined_out)
    slicelist = sorted(os.listdir(dir_multichannel_list[0]))
    whole_body_h, whole_body_w = cv2.imread(dir_multichannel_list[0] + slicelist[0], -1).shape
    whole_body_d = len(slicelist)
    print(whole_body_h, whole_body_w, whole_body_d)

    for z in range(len(slicelist)):
        cur_slice = np.zeros((whole_body_h, whole_body_w), dtype=np.float32)
        for channel in dir_multichannel_list:
            cur_channel = cv2.imread(channel + slicelist[z], -1)
            cur_slice = cur_slice + cur_channel
            
        tiff = TIFFimage(cur_slice, description='')
        tiff.write_file(f"{path_combined_out}/slice_{z:04}.tif", compression='lzw', verbose=False) 

# Root of whole-body image data    
dir_wholebody_data            = ""
organ_name_list               = ['head', 'heart', 'lung', 'liver', 'spleen', 'kideny']
path_bbs_pickle               = os.path.join(dir_wholebody_data, "bbs.pickledump")
path_wholebody_raw            = os.path.join(dir_wholebody_data, "C01", "")

f_bbs = open(path_bbs_pickle, 'rb')
bbs=pickle.load(f_bbs)
for cur_organ_name in organ_name_list:
    
    path_organ_density_nifti = os.path.join(dir_wholebody_data, "organ_results", f"organ_{cur_organ_name}_crop", f"organ_{cur_organ_name}_contrast_density_norm.nii.gz")
    path_map_organ_density_to_wholebody = os.path.join(dir_wholebody_data, "organ_results", f"organ_{cur_organ_name}_crop", f"{cur_organ_name}_contrast_den_in_wholebody", "")

    bb_cur_organ = bbs[cur_organ_name]
    print(bb_cur_organ)
    perorgan_den_to_wholebody(path_density_nifti=path_organ_density_nifti, path_raw=path_wholebody_raw,
                              bb=bb_cur_organ, path_whole_body_out=path_map_organ_density_to_wholebody)


dir_multiden_list = [os.path.join(dir_wholebody_data,  "organ_results", f"organ_{cur_organ_name}_crop", f"{cur_organ_name}_contrast_den_in_wholebody", "")
                        for cur_organ_name in organ_name_list]
path_combined_out = os.path.join(dir_wholebody_data, "density_combined_wholebody")


combine_multichannel(dir_multichannel_list=dir_multiden_list, path_combined_out=path_combined_out) 