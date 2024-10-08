import os
import datetime
import numpy as np
import cv2
import imageio as io
from utils import filehandling

def rebuild_organ_seg_from_patch(path_patches_pred, path_patch_region, path_organmask_slice, path_output, path_output_nifti, organ_name, organ_key, threshold=0.5):
    """
    Get the prediction of whole organ by assembling prediction of organ patches
    
    path_patches_pred: path of segmentation predictions of organ patches
    path_patch_region: pickle file storing patch coordinates that was generated before in the patch cropping step
    path_organmask_slice: path of organ mask image sequence 
    path_output: path to save outpout, including organ segmentation image sequence, and maximum intensity projection of whole organ segmentation
    path_output_nifti: path to organ segmentation result as nifti file
    organ_name: the name of the organ
    threshold: threshold for obtaining binary segmentation results. The range is 0-1. The default is 0.5. 
    """
    if not os.path.exists(path_out):
        os.mkdir(path_out)
    path_slice      = f"{path_out}/TIFF_pred_norm/"
    if not os.path.exists(path_slice):
        os.mkdir(path_slice)
    path_slice      += f"{organ_name}_TIFF_"
    region          = filehandling.pload(path_patch_region)

    x, y, z = region["dataset"]["bounding_box"][:-1]

    volume = np.zeros((x + 100, y + 100, z + 50),dtype=np.float16)
    volume_overlap = np.zeros((x + 100, y + 100, z + 50), dtype=np.uint8)

    len_p  = len(os.listdir(path_patches_pred))

    print(f"{datetime.datetime.now()} Loading patches...")
    for i, patch_item in enumerate(os.listdir(path_patches_pred)):
        if not patch_item.endswith('.nii.gz'):
            continue
        patch                   = filehandling.readNifti(os.path.join(path_patches_pred, patch_item))
        patch                   = patch.astype(np.uint8)
        patch_id                = int(patch_item.replace("patchvolume_","").replace(".nii.gz","")) 
        try:
            off_x, off_y, off_z     = region["patches"][patch_id]["offset"]
            size_x, size_y, size_z  = region["patches"][patch_id]["boundingbox"]

            print(f"{datetime.datetime.now()} Processing\t {patch_item}\t\t\
                    ({i}/{len_p})\t\
                    [{off_x}:{off_x+size_x},\
                    {off_y}:{off_y+size_y},\
                    {off_z}:{off_z+size_z}]\
                    \t{np.sum(patch)}")

            volume[off_x:off_x+size_x, off_y:off_y + size_y, off_z:off_z + size_z] += patch
            volume_overlap[off_x:off_x+size_x, off_y:off_y + size_y, off_z:off_z + size_z] += np.ones(patch.shape, dtype=np.uint8)
        except IndexError as ie:
            print(f"\Index Error at {patch_id}:\t{ie}")
        except ValueError as ve:
            print(f"\nValue Error at {patch_item} shape {patch.shape}:\n {ve}")
            print(region["patches"][patch_id])
            exit()

    volume_overlap[volume_overlap==0]=1
    volume=volume/volume_overlap
    del volume_overlap
    print(f"\n{datetime.datetime.now()} Done; Sum: {np.sum(volume)} Max: {np.amax(volume)}")
    volume = volume[:-100,:-100,:-50]
    print(f"{datetime.datetime.now()} Setting everything to 1")
    volume[volume>threshold]=1
    print(f"{datetime.datetime.now()} Writing TIFF")
    volume = volume.astype(np.uint8)
    for i in range(volume.shape[-1]):
        img_organmask = cv2.imread(os.path.join(path_organmask_slice, sorted(os.listdir(path_organmask_slice))[i]), -1)
        img_organmask = np.squeeze(img_organmask)
        img_organmask[img_organmask!=organ_key]=0
        img_organmask[img_organmask==organ_key]=1
        volume[:, :, i] = volume[:, :, i] * img_organmask
        io.imwrite(f"{path_slice}{i:04}.tif", volume[:, :, i])

    print(f"{datetime.datetime.now()} Calculating MIP")
    mip = np.max(volume, axis=2)
    print(f"{datetime.datetime.now()} Writing TIFF MIP {np.amax(mip)} {np.sum(mip)}")
    io.imwrite(f"{path_out}/{organ_name}_norm_pred_MIP.tif", mip)

    print(f"{datetime.datetime.now()} Writing Volume")
    filehandling.writeNifti(path_nifti_out, volume)

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

path_region          = os.path.join(dir_wholebody_data, "organ_results", f"organ_{cur_organ_name}_crop", "sync_C01region.pickledump")
print(path_region)
path_organmask_slice = os.path.join(dir_wholebody_data, "organ_results", f"organ_{cur_organ_name}_mask")
path_patches_pred    = os.path.join(dir_wholebody_data, "organ_results", f"organ_{cur_organ_name}_crop", "local_C01_norm_infer")
path_out             = os.path.join(dir_wholebody_data, "organ_results", f"organ_{cur_organ_name}_crop")
path_nifti_out       = os.path.join(dir_wholebody_data, "organ_results", f"organ_{cur_organ_name}_crop", f"organ_{cur_organ_name}_pred.nii.gz")

rebuild_organ_seg_from_patch(path_patches_pred=path_patches_pred, path_patch_region=path_region, path_organmask_slice = path_organmask_slice,
                             path_output=path_out, path_output_nifti=path_nifti_out, organ_name=cur_organ_name, organ_key=cur_organ_key)
