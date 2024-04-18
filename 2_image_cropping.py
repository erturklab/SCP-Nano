import os
from utils import cut_volume_single_tiff

# Root of whole-body image data  
dir_wholebody_data = ""
# name of organ to crop into patches 
organ_name = "" # lung, liver, heart, head, spleen, kidney

# set the patch size and overlap size between patches
patch_size = [300, 300, 300] 
patch_overlap = [30, 30, 30]

parameters = {}
parameters["file_format"] = "Nifti"
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
parameters['partitioning']['safe_cache'] = False
parameters['advanced'] = {}
parameters['advanced']['empty_patches'] = [] 
cut_volume_single_tiff.cut_volume(parameters)

