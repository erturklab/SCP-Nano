import sys
import filehandling
import os
import numpy as np
import cv2
import multiprocessing
import functools
import datetime
from libtiff import TIFF
import shutil
from PIL import Image


def imresize(arr, factor):
    size = tuple([int(i * factor) for i in arr.shape])
    return np.array(Image.fromarray(arr).resize(size)).T

def save_patch(region, bitdepth, file_format, patch, multiprocessing="None"):
    before = datetime.datetime.now()
    bb = patch['boundingbox']
    patch['volume'] = np.zeros((bb[0],bb[1],bb[2]), 'uint'+str(bitdepth))
    flist = sorted(os.listdir(region['dataset']['sourcefolder'] + region['dataset']['channelname']))
    if multiprocessing != "None":
        print('Patch ' + str(patch['id']) + ' was taken by ' + multiprocessing.current_process().name)
    for z in range(0,bb[2]):
        file = flist[patch['offset'][2] + z]
        image = cv2.imread(region['dataset']['sourcefolder'] + region['dataset']['channelname'] + '/' + file, 2) # '2' forces cv2 to keep original bitdepth
        y0 = patch['offset'][0]
        x0 = patch['offset'][1]
        image_cut = image[y0:y0+bb[0],x0:x0+bb[1]]
        patch['volume'][:,:,z] = image_cut

    filename = 'patchvolume_' + str(patch['id'])
    croppedfsize = bb[0].astype(np.float) * bb[1] * bb[2] * bitdepth /(8*1024**3)
    if not os.path.exists(region['dataset']['localfolder']):
        os.makedirs(region['dataset']['localfolder'])
    if(file_format == 'TIFF' and croppedfsize<4):
        filehandling.writeFijiTIFF(region['dataset']['localfolder'] + filename + '.tif', patch['volume'])
        print('\nPatch volume saved as Fiji-readable TIFF.')
    else:
        filehandling.writeNifti(region['dataset']['localfolder'] + filename + '.nii.gz', patch['volume'])
        print('\nPatch volume saved as Nifti file.')
        if(file_format == 'TIFF'):
            print('File had to be saved as Nifti instead of TIFF as its size exceeds 4GB, the maximum for TIFFs')
    after = datetime.datetime.now()
    delta = (after-before).total_seconds()
    print("Cutting took " + str(delta) + " seconds.")
    
def insert_array(big_arr, small_arr, x_offset, y_offset):                                                                                                                                                          
    x, y = x_offset, y_offset                                                                                                                                                                                      
    ysize, xsize = small_arr.shape                                                                                                                                                                                 
    xmax, ymax = (x + xsize), (y + ysize)                                                                                                                                                                          
    big_arr[y:ymax, x:xmax] += small_arr                                                                                                                                                                           
    return big_arr  
                
def slice_zdim(parameters, zdim, region, file_format,save_cache=False):
    z_offset = zdim[0]["offset"][2]
    z_range = zdim[0]["offset"][2] + zdim[0]["boundingbox"][2]
    tiffs_to_load = range(z_offset,z_range)
    bitdepth = region['dataset']['bit_depth']
    flist = sorted(os.listdir(region['dataset']['sourcefolder'] + region['dataset']['channelname']))
    zdim_r = [x for x in zdim if not x["id"] in [pid[5:] for pid in os.listdir(parameters['dataset']['cachefolder'])] or str(x["id"]) in parameters['advanced']['empty_patches']]
    print("Slicing tiffs...")

    for tiff in tiffs_to_load:
        # read image for every z-depth
        before = datetime.datetime.now()
        file_name = region['dataset']['sourcefolder'] + region['dataset']['channelname'] + '/' + flist[tiff]
        print(file_name)
        image = cv2.imread(file_name, 2) # '2' forces cv2 to keep original bitdepth
        print(image.shape, image.dtype)
        for patch in zdim_r:
            bb = patch["boundingbox"]
            #z = tiff
            y0 = patch['offset'][0]
            x0 = patch['offset'][1]
            image_cut = np.zeros(region['partitioning']['patch_size'][:2])
            image_cut = insert_array(image_cut, image[y0:y0+bb[0],x0:x0+bb[1]], 0, 0)
            path = os.path.join(parameters['dataset']['cachefolder'], "patch" + str(patch["id"]))

            #l_  = [i_ == 200 for i_ in image_cut.shape]      
            if not os.path.exists(path):
                os.makedirs(path)
            tiff_out = TIFF.open(os.path.join(path, "tiffslice_" + str(tiffs_to_load.index(tiff)) + ".tif"),"a")
            try:
                tiff_out.write_image(image_cut)
            except AssertionError as aex:
                print("\nError {}".format(aex))
                print("Tiff {} \nPatch {} \nShape {}".format(tiff, patch, image_cut.shape))
                print("image shape {} x0, y0 {},{} bb[0],bb[1] {}{}".format(image.shape, x0, y0, bb[0], bb[1]))
                print("Min {} Max {} Mean {}".format(np.amin(image_cut), np.amax(image_cut), np.mean(image_cut)))
                exit()
            tiff_out.close()
        after = datetime.datetime.now()
        delta = (after-before).total_seconds()
        sys.stdout.flush()
        print("Cut tiff (" + str(tiff) + ")" + str(tiffs_to_load.index(tiff)) + " of " + str(len(tiffs_to_load))+". Cutting took " + str(delta) + " seconds. Overall " + str(tiffs_to_load.index(tiff)/len(tiffs_to_load)) + "%")
    print("Sliced tiffs.")
    region = save_slice(parameters, zdim, bitdepth, file_format, region, save_cache)
    return region

def update_min_max(temp, general):
    if temp[0] < general[0]:
        general[0] = temp[0]
    if temp[1] > general[1]:
        general[1] = temp[1]
    return general
    

def save_slice(parameters, zdim, bitdepth, file_format, region, save_cache=True):
    print("Saving as " + file_format + "...")

    min_max = [region["dataset"]["min_value"], region["dataset"]["max_value"]]

    for patch in zdim:
      before = datetime.datetime.now()
      in_path = os.path.join(parameters['dataset']['cachefolder'], "patch" + str(patch["id"]))
      out_path = os.path.join(parameters['dataset']['localfolder'], 'patchvolume_' + str(patch['id']))
      sort_tifs(parameters, patch)
      if not os.path.exists(parameters['dataset']['localfolder']):
          os.makedirs(parameters['dataset']['localfolder'])

      volume = np.zeros(patch["boundingbox"])
      for i,file_name in enumerate(sorted(os.listdir(in_path))):
          image_path = os.path.join(in_path, file_name)
          im = cv2.imread(image_path, 2)
          print(i)
          print(in_path)
          print(file_name)
          print(im.shape)
          volume[:,:,i] = im
      temp = (np.amin(volume), np.amax(volume))
      min_max = update_min_max(temp, min_max)
      if file_format == "Nifti":
          filehandling.writeNifti(out_path + ".nii", volume)
      else:
          filehandling.writeFijiTIFF(out_path + ".tif", volume)
      after = datetime.datetime.now()
      delta = (after-before).total_seconds()
      print("Saved patch " + str(patch["id"]) + "("+ str(zdim.index(patch)) +" of " + str(len(zdim)) +")"+ " in " + str(delta) + " seconds.")
      if not save_cache:
          shutil.rmtree(in_path)
          print("Removed cache file for patch " + str(patch["id"]))
    region["dataset"]["min_value"] = min_max[0]
    region["dataset"]["max_value"] = min_max[1]
    return region

#helper function to name files with leading zeros
def sort_tifs(parameters, patch):
    in_path = os.path.join(parameters['dataset']['cachefolder'], "patch" + str(patch["id"]))
    list_tifs = os.listdir(in_path)
    z_dim = parameters['partitioning']['patch_size'][2]
    for tif in list_tifs:
        tif_number = tif[10:-4]
        while len(tif_number) < len(str(z_dim)):
            tif_number = "0" + tif_number
        tif_new = tif[0:10] + tif_number + tif[-4:]
        path_old = os.path.join(in_path, tif)
        path_new = os.path.join(in_path, tif_new)
        os.rename((path_old),(path_new))

def cut_volume(parameters):
    #%% Define parameters
    print(parameters['dataset']['syncfolder'])
    num_cores_to_use = multiprocessing.cpu_count()-1
    file_format = parameters["file_format"] # choose from Nifti or TIFF
    region = {}
    region['name'] = parameters["region_name"]
    thumbnail_ds = 0.1 # Downsampling for thumbnails of patches
    # Parameters: data set
    region['dataset'] =  parameters["dataset"]
    # Parameters: cropping & dicing
    region['partitioning'] = parameters['partitioning']

    #%% Set up folders
    if not os.path.exists(parameters['dataset']['localfolder']):
        os.makedirs(parameters['dataset']['localfolder'])
    if not os.path.exists(parameters['dataset']['syncfolder']):
        os.makedirs(parameters['dataset']['syncfolder'])
    if not os.path.exists(parameters['dataset']['cachefolder']):
        os.makedirs(parameters['dataset']['cachefolder'])

    #%% Set up patches

    # Compute further parameters
    (fsize, dims, bitdepth) = filehandling.estimateVolSize(region['dataset']['sourcefolder'],region['dataset']['channelname'])
    region['dataset']['file_size'] = fsize
    region['dataset']['bounding_box'] = dims
    region['dataset']['bit_depth'] = bitdepth
    region['dataset']['min_value'] = 1
    region['dataset']['max_value'] = 0
    patch_size = np.asarray(region['partitioning']['patch_size'],np.uint16)
    patch_overlap = np.asarray(region['partitioning']['patch_overlap'],np.uint16)
    n_patches = np.asarray((np.asarray(region['partitioning']['cropping_boundingbox']) - np.asarray(region['partitioning']['cropping_offset']) - patch_overlap)/ (patch_size - patch_overlap),np.uint8)
    print('Number of patches: ',n_patches )
    region['partitioning']['patches_per_dim'] = n_patches
    # Initialize patches
    region['patches'] = []
    patch_id = 0
    for ny in range(0,n_patches[0]):
        for nx in range(0,n_patches[1]):
            for nz in range(0,n_patches[2]):
                patch = {}
                patch['id'] = patch_id
                patch['patchstep'] = [ny, nx, nz]
                patch['offset'] = region['partitioning']['cropping_offset'] + (patch_size - patch_overlap) * patch['patchstep']
                patch['boundingbox'] = patch_size
                region['patches'].append(patch)
                patch_id += 1
    # Find patches that still need to be cut
    
    remaining_patches = []
    for patch in region['patches']:
        if not str(patch['id']) in parameters['advanced']['empty_patches']:
            if(os.path.isfile(os.path.join(region['dataset']['localfolder'], 'patchvolume_' + str(patch['id']) + '.nii'))==False and file_format == 'Nifti'):
                remaining_patches.append(patch)
            elif(os.path.isfile(os.path.join(region['dataset']['localfolder'], 'patchvolume_' + str(patch['id']) + '.tif'))==False and file_format == 'TIFF'):
                remaining_patches.append(patch)

    region_path_ = os.path.join(region['dataset']['syncfolder'], 'region')
    filehandling.psave(region_path_, region) # saving initial region file (will be updated later on)| beginn parameters with ROOTP + 
    print('Patches initialized, now segmenting volume')

    #%% Execution
    # Cut volume into patches and save them to file    
    if parameters['multiprocessing']:
        print("Using multiprocessing to cut volume to patches...")
        pool = multiprocessing.Pool(processes=num_cores_to_use)
        pool.map(functools.partial(save_patch,
                                region=region, 
                                bitdepth=bitdepth, 
                                multiprocessing=multiprocessing, 
                                file_format=file_format), 
                            remaining_patches)
        pool.close()
    else:
        single_tiff_list = []
        unique_set = set()
        for x in remaining_patches:
            unique_set.add(x["offset"][2])
        unique_set = sorted(unique_set)
        for z_dim in unique_set:
            z_dim_list = []
            for x in remaining_patches:
                if x["offset"][2] == z_dim: z_dim_list.append(x)
            single_tiff_list.append(z_dim_list)

        print("Cutting " + str(len(single_tiff_list)) + " Z dimensions...")
        print(single_tiff_list)
        for z_dim in single_tiff_list:
            print("Slicing z dimension " + str(single_tiff_list.index(z_dim)))
            before = datetime.datetime.now()
            region = slice_zdim(parameters, z_dim, region,"Nifti",save_cache=parameters['partitioning']['safe_cache'])
            after = datetime.datetime.now()
            delta = after - before
            total = delta * len(single_tiff_list)
            print("Patch took " + str(delta) + "seconds. Overall remaining dataset (estimated): " + str(total))
    
    print('Entire volume cut to patches and saved to file. Creating thumbnails...')
    
    # Create thumbnails for all patches
    bbx = int((region['partitioning']['cropping_boundingbox'][0] - parameters['partitioning']['cropping_offset'][0])*thumbnail_ds)
    bby = int((region['partitioning']['cropping_boundingbox'][1] - parameters['partitioning']['cropping_offset'][1])*thumbnail_ds)
    bbz = int((region['partitioning']['cropping_boundingbox'][2] - parameters['partitioning']['cropping_offset'][2])*thumbnail_ds)
    mpx = np.zeros((bby,bbz,region['partitioning']['patches_per_dim'][0]),np.float32)
    mpy = np.zeros((bbx,bbz,region['partitioning']['patches_per_dim'][1]),np.float32)
    mpz = np.zeros((bbx,bby,region['partitioning']['patches_per_dim'][2]),np.float32)
    dx = int((region['partitioning']['patch_size'][0] - region['partitioning']['patch_overlap'][0])*thumbnail_ds)
    dy = int((region['partitioning']['patch_size'][1] - region['partitioning']['patch_overlap'][1])*thumbnail_ds)
    dz = int((region['partitioning']['patch_size'][2] - region['partitioning']['patch_overlap'][2])*thumbnail_ds)
    print('ds',dx , dy ,  dz)
    print('bbs',bbx,bby,bbz)
    print('shapes of mps:',  mpx.shape, mpy.shape, mpz.shape)
    for patch in region['patches']:
        print('Creating thumbnail for patch #' + str(patch['id']))
        if not str(patch['id']) in parameters['advanced']['empty_patches']:
            path_v = os.path.join(region['dataset']['localfolder'], 'patchvolume_' + str(patch['id']))
            pvol = filehandling.readNifti(path_v)
            pvol = pvol[0:int(dx/thumbnail_ds),0:int(dy/thumbnail_ds),0:int(dz/thumbnail_ds)]
            s = patch['patchstep']
            print('ds',dx,dy,dz)
            print(s)
            print(imresize(np.max(pvol,2),thumbnail_ds).shape)
            mpz[s[0]*dx:(s[0]+1)*dx, s[1]*dy:(s[1]+1)*dy, s[2]] = imresize(np.max(pvol,2),thumbnail_ds, mode='F')
            mpz[s[0]*dx,:,s[2]] = mpz[:,s[1]*dy,s[2]] = 0

    # Update and save region overview to file
    region['thumbnails'] = {}
    region['thumbnails']['downsampling'] = thumbnail_ds
    region['thumbnails']['MaxProjections_Z'] = mpz
    print('Saving region file to:', region['dataset']['syncfolder'] + 'region')
    
    filehandling.psave(region['dataset']['syncfolder'] + 'region', region)
    print('Region file updated with thumbnails.')
    print('Done.')









