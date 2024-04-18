import os
import datetime
import numpy as np
from libtiff import TIFFimage
from utils import filehandling

def rebuild_organ_seg_from_patch(path_patches_pred, path_patch_region, path_output, path_output_nifti, organ_name, threshold=0.5):
    """
    Get the prediction of whole organ by assembling prediction of organ patches
    
    path_patches_pred: path of segmentation predictions of organ patches
    path_patch_region: pickle file storing patch coordinates that was generated before in the patch cropping step
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
    volume = volume.astype(np.uint8)

    print(f"{datetime.datetime.now()} Calculating MIP")
    mip = np.max(volume, axis=2)
    print(f"{datetime.datetime.now()} Writing TIFF MIP {np.amax(mip)} {np.sum(mip)}")
    tiff = TIFFimage(mip, description='')
    tiff.write_file(f"{path_out}/{organ_name}_norm_pred_MIP.tif", compression='lzw', verbose=False)

    print(f"{datetime.datetime.now()} Writing TIFF")
    for i in range(volume.shape[-1]):
        tiff = TIFFimage(volume[:,:,i], description='')
        tiff.write_file(f"{path_slice}{i:04}.tif", compression='lzw', verbose=False)

    print(f"{datetime.datetime.now()} Writing Volume")
    filehandling.writeNifti(path_nifti_out, volume)

# Root of whole-body image data    
dir_wholebody_data = ""
# name of organ to crop into patches 
organ_name = ""

path_region          = os.path.join(dir_wholebody_data, "organ_results", f"organ_{organ_name}_crop", "sync_C01region.pickledump")
path_patches_pred    = os.path.join(dir_wholebody_data, "organ_results", f"organ_{organ_name}_crop", "local_C01_norm_infer")
path_out             = os.path.join(dir_wholebody_data, "organ_results", f"organ_{organ_name}_crop")
path_nifti_out       = os.path.join(dir_wholebody_data, "organ_results", f"organ_{organ_name}_crop", f"organ_{organ_name}_pred.nii.gz")

rebuild_organ_seg_from_patch(path_patches_pred=path_patches_pred, path_patch_region=path_region,
                             path_output=path_out, path_output_nifti=path_nifti_out, organ_name=organ_name)
