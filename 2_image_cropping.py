import os
from utils import cut_volume_single_tiff

# Root of whole-body image data  
dir_wholebody_data = ""
# name of organ to crop into patches 
organ_name = "" # lung, liver, heart, head, spleen, kidney
# set the patch size and overlap size between patches
# set the cropping start and end coordinates, specifying a bounding box region to do the cropping

# !!!Setting requirement: cropping_end - cropping_start - patch_overlap = n * (patch_size-patch_overlap), n is an iteger
# The order is [x_dim, y_dim, z_dim]
patch_size = [418, 460, 600] 
patch_overlap = [19, 15, 0]
cropping_start = [0, 0, 0]
cropping_end = [7600, 1350, 600]


parameters = {}
parameters["file_format"] = "Nifti"
parameters["region_name"] = "signal" 
parameters["thumbnail_downsampling"] = 0.1 
parameters["dataset"] = {}
parameters["dataset"]["channelname"] = ""
parameters['dataset']['sourcefolder'] = os.path.join(dir_wholebody_data, "organ_results", f"organ_{organ_name}_raw", "")
parameters['dataset']['localfolder'] = os.path.join(dir_wholebody_data, "organ_results", f"organ_{organ_name}_crop", "local_C01")
parameters['dataset']['syncfolder'] =  os.path.join(dir_wholebody_data, "organ_results", f"organ_{organ_name}_crop", "sync_C01")
parameters['dataset']['cachefolder'] = os.path.join(dir_wholebody_data, "organ_results", f"organ_{organ_name}_crop", "cache_C01")
parameters['dataset']['downsampling'] = 'None'
parameters['partitioning'] = {}
parameters['partitioning']['patch_size'] = patch_size
parameters['partitioning']['patch_overlap'] = patch_overlap
parameters['partitioning']['cropping_offset'] = cropping_start
parameters['partitioning']['cropping_boundingbox'] = cropping_end
parameters['partitioning']['safe_cache'] = False
parameters['multiprocessing'] = False
parameters['advanced'] = {}
parameters['advanced']['empty_patches'] = [] 
cut_volume_single_tiff.cut_volume(parameters)

