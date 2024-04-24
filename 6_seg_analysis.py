import cv2
import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import cc3d
import nibabel as nib
import math

def intensity_stat_forebackground(path_organpred_slice, path_organraw_slice, path_output, organ_key, threshold = 0.5):
    """
    Statistical intensity distributions of segmented regions (foreground) and the rest (background) in one organ,
    Return the background intensity mean and foreground intensity mean

    path_organpred_slice: path of organ segmentation image sequence
    path_organraw_slice: path of organ raw image sequence
    path_output: path to save intensity histgram
    organ_key: label value of the organ  
    threshold: threshold to apply on prediction. Default is 0.5. The range is 0-1.
    """
    if not os.path.exists(path_output):
        os.mkdir(path_output)

    background_voxel_intensity = np.array([])
    foreground_voxel_intensity = np.array([])
    for z, z_slice in enumerate(sorted(os.listdir(path_organpred_slice))):
        print(z_slice)
        print(sorted(os.listdir(path_organraw_slice))[z])
        img_seg = cv2.imread(path_organpred_slice + z_slice, -1)
        img_seg = np.squeeze(img_seg)

        img_raw = cv2.imread(path_organraw_slice + sorted(os.listdir(path_organraw_slice))[z], -1)
        img_raw = np.squeeze(img_raw)
            
        print('intensity value of segmentation:', np.min(img_seg), np.max(img_seg))
        print('intensity value of raw image:', np.min(img_raw), np.max(img_raw))

        if threshold != -1:
            img_seg[img_seg  < threshold] = 0
            img_seg[img_seg  > threshold] = 1

        foreground_voxel_value = img_raw * img_seg
            
        foreground_voxel_intensity = np.concatenate([foreground_voxel_intensity, foreground_voxel_value[foreground_voxel_value>0]])

        img_seg_pro = np.copy(img_seg)
        img_seg_pro = scipy.ndimage.binary_dilation(img_seg_pro).astype(img_seg_pro.dtype)
        background_seg = 1 - img_seg_pro
        background_voxel_value = img_raw * background_seg
        background_voxel_intensity = np.concatenate([background_voxel_intensity, background_voxel_value[background_voxel_value>0]])
        
    if len(foreground_voxel_intensity)==0:
        background_mean = None
        foreground_mean = None
    else:  
        print('intensity range of foreground voxel:', np.min(foreground_voxel_intensity), np.max(foreground_voxel_intensity))
        print('intensity range of background voxel:', np.min(background_voxel_intensity), np.max(background_voxel_intensity))
      
        sorted_foreground_intensity = np.sort(foreground_voxel_intensity)
        print('95 percent of foreground intensity thresh:', sorted_foreground_intensity[int(0.05*len(sorted_foreground_intensity))])
        sorted_background_intensity = np.sort(background_voxel_intensity)
        print('95 percent of background intensity thresh:', sorted_background_intensity[int(0.95*len(sorted_background_intensity))])
        plt.figure(1)
        plt.hist(foreground_voxel_intensity, bins=100)
        plt.savefig(path_output +  "histogram_foreground.png")
        plt.figure(2)
        plt.hist(background_voxel_intensity, bins=100)
        plt.savefig(path_output +  "histogram_background.png")
        
        background_mean = np.mean(sorted_background_intensity[int(0.05*len(sorted_background_intensity)):int(0.95*len(sorted_background_intensity))])
        foreground_mean = np.mean(sorted_foreground_intensity[int(0.05*len(sorted_foreground_intensity)):int(0.95*len(sorted_foreground_intensity))])
        
    print("estimated mean intensity value of background:", background_mean)
    print("estimated mean intensity value of foreground:", foreground_mean)
        
    return background_mean, foreground_mean


def intensity_contrast_signal(path_organpred_slice, path_organraw_slice, path_output, organ_key, background_mean, threshold=0.5):
    """
    Quantification of segmented regions (foreground) in an organ,
    Return the relative intensity contrast sum of segmented regions compared to background.

    path_organpred_slice: path of organ segmentation image sequence
    path_organraw_slice: path of organ raw image sequence
    path_output: path to save intensity histgram
    organ_key: label value of the organ  
    background_mean: the mean of background intensity in an organ
    threshold: threshold to apply on prediction. Default is 0.5. The range is 0-1.
    """
    if not os.path.exists(path_output):
        os.mkdir(path_output)

    relative_contrast = np.array([])
    for z, z_slice in enumerate(sorted(os.listdir(path_organpred_slice))):
        print(z_slice)
        print(sorted(os.listdir(path_organraw_slice))[z])
        img_seg = cv2.imread(path_organpred_slice + z_slice, -1)
        img_seg = np.squeeze(img_seg)

        img_raw = cv2.imread(path_organraw_slice + sorted(os.listdir(path_organraw_slice))[z], -1)
        img_raw = np.squeeze(img_raw)
 
        print('intensity value of raw image:', np.min(img_raw), np.max(img_raw))

        if threshold != -1:
            img_seg[img_seg  < threshold] = 0
            img_seg[img_seg  > threshold] = 1


        foreground_voxel_value = img_raw * img_seg

        positive_foreground_voxel_value = foreground_voxel_value[foreground_voxel_value>0]
        foreground_relative_contrast = (positive_foreground_voxel_value-background_mean) / background_mean # Weber contrast
        relative_contrast = np.concatenate([relative_contrast, foreground_relative_contrast])
        
    print('foreground relative contrast range:', np.min(relative_contrast), np.max(relative_contrast))
    plt.figure()
    plt.hist(relative_contrast, bins = 200)
    plt.savefig(path_output +  "relative_contrast.png")
    print("estimated foreground relative contrast sum:", np.sum(relative_contrast[relative_contrast>0]))
    return np.sum(relative_contrast[relative_contrast>0])



def component_filter(volume, component_thresh=3):
    """
    filter small segmented components by a threshold of the size

    volume: a binary numpy array
    component_thresh: threshold for filtering small component by the size
    
    """
    labels = cc3d.connected_components(volume)
    
    stats = cc3d.statistics(labels)
    cnt = np.array(stats["voxel_counts"])    
    components_to_remove =np.where(cnt<component_thresh)[0]

    for c in range(len(components_to_remove)):

        labels[labels==components_to_remove[c]]=0
    labels[labels>0]=1
    
    return labels.astype(volume.dtype)

def generate_heatmap_3d(path_pred, path_heatmap, path_raw, window_size=[16, 16, 4], background_mean=0):
    """
    Generate the density heatmap map of segmented nanoparticles throughout whole organ

    path_pred: path of the segmentation nifti file
    path_heatmap: path to save the density heatmap
    path_raw: path of the corresponding raw image nifti file
    window_size: we use the sliding window strategy to compute density map. Define a window size, 
                 then relative intensity contrast sum of the segmented nanoparticles inside this window region will be calculated as the density of this local region

    background_mean: the mean of background intensity in an organ, used when calculating relative intensity contrast

    """
    assert path_raw != "", "Please specify a raw path!"
    raw_vol = nib.load(path_raw)
    raw_img = raw_vol.get_data()
    affine = raw_vol.affine
    header = raw_vol.header
    del raw_vol
    print(np.min(raw_img), np.max(raw_img))
    
    
    if background_mean == None:
        density_arr = np.zeros_like(raw_img, dtype=np.float32)
        heatmap = nib.Nifti1Image(density_arr, affine, header)
        nib.save(heatmap, path_heatmap)
        return

    pred_vol = nib.load(path_pred)
    pred_arr = pred_vol.get_data()
    del pred_vol
    pred_arr[pred_arr > 0] = 1

    # get window size
    img_shape = list(raw_img.shape)
    print(img_shape)
    if isinstance(window_size, list) or isinstance(window_size, tuple):
        window_size = window_size
    elif isinstance(window_size, int):
        window_size = [window_size, window_size, window_size]
    elif isinstance(window_size, float):
        window_size = [int(i*window_size) for i in img_shape]
    else:
        raise ValueError("invalid windowsize")
    print(window_size)

    # get density array by sliding window
    grid_per_dim = [math.ceil(img_shape[i] / window_size[i]) for i in range(len(img_shape))]
    print(grid_per_dim)
    density_arr = np.zeros_like(raw_img, dtype=np.float32)
    # struct = scipy.ndimage.generate_binary_structure(3, 1)
    for id_x in range(grid_per_dim[0]):
        start_x = id_x * window_size[0]
        end_x = start_x + window_size[0]
        if end_x > img_shape[0]:
            end_x = img_shape[0]
        for id_y in range(grid_per_dim[1]):
            start_y = id_y * window_size[1]
            end_y = start_y + window_size[1]
            if end_y > img_shape[1]:
                end_y = img_shape[1]
            for id_z in range(grid_per_dim[2]):
                start_z = id_z * window_size[2]
                end_z = start_z + window_size[2]
                if end_z > img_shape[2]:
                    end_z = img_shape[2]

                pred_in_window = pred_arr[start_x:end_x, start_y:end_y, start_z:end_z]
                raw_in_window = raw_img[start_x:end_x, start_y:end_y, start_z:end_z]
                pred_in_window_pro = component_filter(pred_in_window, component_thresh=3)
                if np.sum(pred_in_window_pro)==0:
                    density_arr[start_x:end_x, start_y:end_y, start_z:end_z] = 0
                else:
                    masked_voxel_value = raw_in_window * pred_in_window_pro
                    foreground_voxel_value = masked_voxel_value[masked_voxel_value>0]
                    foreground_relative_contrast = (foreground_voxel_value-background_mean) / np.float32(background_mean) # Weber contrast
                    if len(foreground_relative_contrast[foreground_relative_contrast>0])==0:
                        density_arr[start_x:end_x, start_y:end_y, start_z:end_z] = 0
                    else:
                        density_arr[start_x:end_x, start_y:end_y, start_z:end_z] = np.mean(foreground_relative_contrast[foreground_relative_contrast>0])
    del pred_arr
    raw_img[raw_img>0]=1
    raw_img[raw_img<0]=0
    density_arr = density_arr * raw_img
    del raw_img

    density_range = np.max(density_arr)
    print(density_range)
    density_arr = scipy.ndimage.filters.gaussian_filter(density_arr, [window_size[0], window_size[1], window_size[2]], mode='constant')
    density_arr = (density_arr - np.min(density_arr)) / (np.max(density_arr) - np.min(density_arr))
    density_arr = density_range * density_arr
    heatmap = nib.Nifti1Image(density_arr, affine, header)
    heatmap.set_data_dtype(np.float32)
    nib.save(heatmap, path_heatmap)


# Root of whole-body image data    
dir_wholebody_data = ""
# name of organ to crop into patches 
cur_organ_name = ""

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
cur_organ_key = int(organ_keys_list[organ_names_list.index(cur_organ_name)])

path_organpred_slice = os.path.join(dir_wholebody_data, "organ_results", f"organ_{cur_organ_name}_crop", "TIFF_pred_norm", "")
path_organraw_slice = os.path.join(dir_wholebody_data, "organ_results", f"organ_{cur_organ_name}_raw", "")
path_output = os.path.join(dir_wholebody_data, "organ_results", f"organ_{cur_organ_name}_crop", "")

# Estimate background intensity mean
background_mean, foreground_mean = intensity_stat_forebackground(path_organpred_slice=path_organpred_slice, path_organraw_slice=path_organraw_slice, 
                                                   path_output=path_output, organ_key=cur_organ_key, threshold = 0.5)

# Quantification of nanoparticle segmentation by relative intensity contrast sum
if foreground_mean == None:
    print("No significant particles are detected")
    print("relative_contrast_sum:", 0)
else:
    relative_contrast_sum = intensity_contrast_signal(path_organpred_slice=path_organpred_slice, path_organraw_slice=path_organraw_slice, 
                                                  path_output=path_output, organ_key=cur_organ_key, background_mean=background_mean, threshold=0.5)
    print("relative_contrast_sum:", relative_contrast_sum)

path_raw_nifti = os.path.join(dir_wholebody_data, "organ_results", f"organ_{organ_name}_raw.nii.gz")
path_pred_nifti = os.path.join(dir_wholebody_data, "organ_results", f"organ_{organ_name}_crop", f"organ_{organ_name}_pred.nii.gz")
path_heatmap_nifti = os.path.join(dir_wholebody_data, "organ_results", f"organ_{organ_name}_crop", f"organ_{organ_name}_contrast_density_norm.nii.gz")
generate_heatmap_3d(path_pred=path_pred_nifti, path_heatmap=path_heatmap_nifti, path_raw=path_raw_nifti, background_mean = background_mean)
