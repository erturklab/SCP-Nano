import os 
import cv2 
import numpy as np
from libtiff import TIFFfile
from libtiff import TIFF
import re
import pickle
import nibabel as nib



def psave(path, variable):
    '''
    psave(path, variable)
    
    Takes a variable (given as string with its name) and saves it to a file as specified in the path.
    The path must at least contain the filename (no file ending needed), and can also include a 
    relative or an absolute folderpath, if the file is not to be saved to the current working directory.
    
    # ToDo: save several variables (e.g. take X args, store them to special DICT, and save to file)
    '''
    if(path.find('.pickledump')==-1):
        path = path + '.pickledump'
    
    folderpath = '/'.join([folder for folder in path.split('/')[0:-1]])
    if(os.path.isdir(folderpath) == False):
        os.makedirs(folderpath) # create folder(s) if missing so far.
    file = open(path, 'wb')
    pickle.dump(variable,file,protocol=4)


def pload(path):
    '''
    variable = pload(path)
    
    Loads a variable from a file that was specified in the path. The path must at least contain the 
    filename (no file ending needed), and can also include a relative or an absolute folderpath, if 
    the file is not to located in the current working directory.
    
    # ToDo: load several variables (e.g. load special DICT from file and return matching entries)
    '''
    if(path.find('.pickledump')==-1):
        path = path + '.pickledump'
    path = path.replace('\\','/')
    cwd = os.getcwd().replace('\\','/')
    if(path[0:2] != cwd[0:2] and path[0:5] != '/mnt/'):
        path = os.path.abspath(cwd + '/' + path) # If relatice path was given, turn into absolute path
    file = open(path, 'rb')
    return pickle.load(file)


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
    n_c = len(folderlist)
    if n_c > 1:
        folder = pathtofolders + '/' + folderlist[0]
    else:
        folder = pathtofolders
    filelist = os.listdir(folder)
    n_z = len(filelist)
    path = folder + '/' + filelist[0]
    print("Path : {}".format(path))
    image = cv2.imread(path,2)
    (n_y,n_x) = image.shape
    bitdepth = int(''.join(filter(str.isdigit, str(image.dtype))))
    size = int(n_y*n_x*n_z*n_c*bitdepth / 8 / 1024**3)
    return (size,(n_y,n_x,n_z,n_c),bitdepth)


def readFijiTIFF(path):
    '''
    (volume, FijiMeta) = readFijiTIFF(path)
        - Reads in 4D TIFF file as saved by Fiji and returns a Numpy volume (y,x,z,c) as well as a readable version of the file's meta data
    
    volume (Numpy array)
        * y: image height
        * x: image width
        * z: image depth (number of slices)
        * c: number of channels
    
    FijiMeta (dict)
        * -- stored within TiffTag 'ImageDescription' --
        * version: Fiji software version
        * nimages: total number of images (z*c)
        * nchannels: number of channels (c)
        * nslices: number of slices (z)
        * nhyperstack: 'true' if 4D volume
        * nmode: 'composite' if Fiji was used to merge channels
        * nloop: 'false'
        * nmin: minimum value for Fiji signal scaling
        * nmax: maximum value in Fiji signal scaling
        * -- stored as standalone TiffTags --
        * ImageLength: image height (y)
        * ImageWidth: image width (x)
        * PhotometricInterpretation: how to interpret pixel values (e.g. 'BlackIsZero')
        * PlanarConfiguration: (unclear)
    '''
    if(path.find('.tif')==-1):
        path = path + '.tif'
    tiff = TIFFfile(path)
    # Read out images and save in Numpy array 
    samples, sample_names = tiff.get_samples()
    outList = []
    for sample in samples:
        outList.append(np.copy(sample)[...,np.newaxis])
    volume = np.concatenate(outList,axis=-1)
    volume = np.rollaxis(volume,0,3)
    
    # Read out Fiji meta data
    template = r'ImageJ=(?P<version>.*)\nimages=(?P<nimages>.*)\nchannels=(?P<nchannels>.*)\nslices=(?P<nslices>.*)\nhyperstack=(?P<nhyperstack>.*)\nmode=(?P<nmode>.*)\nloop=(?P<nloop>.*)\nmin=(?P<nmin>.*)\nmax=(?P<nmax>.*)\n\x00'
    ifd = tiff.get_first_ifd(0)
    FijiMeta_IDb = ifd.get_value('ImageDescription', human=True)
    FijiMeta_ID = FijiMeta_IDb.decode()
    #FijiMeta_PIh = ifd.get_value('PhotometricInterpretation', human=True) --> would give 'BlackIsZero' instead of 1
    FijiMeta_PI = ifd.get_value('PhotometricInterpretation')
    FijiMeta_PC = ifd.get_value('PlanarConfiguration')
    FijiMeta_O = ifd.get_value('Orientation')
    FijiMeta_L = ifd.get_value('ImageLength')
    FijiMeta_W = ifd.get_value('ImageWidth')
    try:
        matches = re.match(template, FijiMeta_ID)
        FijiMeta = {'version': matches.group('version'),
                    'nimages': int(matches.group('nimages')),
                    'nchannels': int(matches.group('nchannels')),
                    'nslices': int(matches.group('nslices')),
                    'nhyperstack': matches.group('nhyperstack'),
                    'nmode': matches.group('nmode'),
                    'nloop': matches.group('nloop'),
                    'nmin': matches.group('nmin'),
                    'nmax': matches.group('nmax'),
                    'PhotometricInterpretation': FijiMeta_PI,
                    'PlanarConfiguration': FijiMeta_PC,
                    'Orientation': FijiMeta_O,
                    'ImageLength': FijiMeta_L,
                    'ImageWidth': FijiMeta_W}
        # Rearrange volume from (y,x,z*c,f=1) to (y,x,z,c)
        if FijiMeta['nimages'] == FijiMeta['nchannels']*FijiMeta['nslices'] == volume.shape[2]:
            volume_old = volume
            volume = np.zeros((volume.shape[0], volume.shape[1], FijiMeta['nslices'], FijiMeta['nchannels']), volume_old.dtype.name)
            for c in range(0,FijiMeta['nchannels']):
                volume[:,:,:,c] = volume_old[:,:,c:FijiMeta['nimages']:FijiMeta['nchannels'],0]
    except:
        FijiMeta = 'Failed to read meta data'
    
    tiff.close()
    return(volume,FijiMeta)


def writeFijiTIFF(path, volume, FijiMeta=None, compression=None):
    '''
    Writes volume (and meta data) to 4D TIFF file that can be read as 'composite' by Fiji. Meta data does not have to be provided - default values will be used and updated to volume size if necessary. If meta data is provided, it needs to be the full dictionary with all values.
    
    volume (3D or 4D Numpy array)
        * y: image height
        * x: image width
        * z: image depth (number of slices)
        * c: number of channels (c=1 for 3D array)
    
    Optional: FijiMeta (dict)
        * -- stored within TiffTag 'ImageDescription' --
        * version: Fiji software version
        * nimages: total number of images (z*c)
        * nchannels: number of channels (c)
        * nslices: numder of slices (z)
        * nhyperstack: 'true' if 4D volume
        * nmode: 'composite' if Fiji was used to merge channels
        * nloop: 'false'
        * nmin: minimum value for Fiji signal scaling
        * nmax: maximum value in Fiji signal scaling
        * -- stored as standalone TiffTags --
        * ImageLength: image height (y)
        * ImageWidth: image width (x)
        * PhotometricInterpretation: how to interpret pixel values (e.g. 'BlackIsZero')
        * PlanarConfiguration: (unclear)
    '''
    
    # Convert 3D volume to 4D volume, if necessary
    if(volume.ndim == 3): volume = np.expand_dims(volume, axis=3)
    if(volume.ndim != 4): raise ValueError("Volume must be 3D or 4D")
    
    # Check/update meta information
    if FijiMeta == None:
        # Use default values
        status = 'No meta information provided; we therefore used default values'
        FijiMeta = {'version': '1.51u',
                    'nimages': volume.shape[2]*volume.shape[3],
                    'nchannels': volume.shape[3],
                    'nslices': volume.shape[2],
                    'nhyperstack': 'true',
                    'nmode': 'composite',
                    'nloop': 'false',
                    'nmin': np.min(volume),
                    'nmax': np.max(volume),
                    'PhotometricInterpretation': 1, # BlackIsZero
                    'PlanarConfiguration': 1, # Chunky
                    'Orientation': 1, # TopLeft
                    'ImageLength': volume.shape[0],
                    'ImageWidth': volume.shape[1]}
    else:
        delta = {'nimages': volume.shape[2]*volume.shape[3],
                 'nchannels': volume.shape[3],
                 'nslices': volume.shape[2],
                 'ImageLength': volume.shape[0],
                 'ImageWidth': volume.shape[1]}
        new = FijiMeta
        if new.update(delta) != FijiMeta: status = 'Had to update meta information to match volume'
        FijiMeta = new
    
    # Encode meta information
    template = 'ImageJ={version}\nimages={nimages}\nchannels={nchannels}\nslices={nslices}\nhyperstack={nhyperstack}\nmode={nmode}\nloop={nloop}\nmin={nmin}\nmax={nmax}\n\x00'
    FijiMeta_ID = template.format(**FijiMeta)
    FijiMeta_IDb = FijiMeta_ID.encode()
    
    # Rearrange volume from (y,x,z,c) to (y,x,z*c,f=1)
    bitdepth = int(''.join(filter(str.isdigit, str(volume.dtype.name))))
    size = volume.shape[0]*volume.shape[1]*FijiMeta['nslices']*FijiMeta['nchannels']*bitdepth / 1024**3
    if size > 4: raise ValueError("Cannot write TIFF files larger than 4GB")
    if FijiMeta['nslices'] == volume.shape[2] and FijiMeta['nchannels'] == volume.shape[3] and FijiMeta['nimages'] == volume.shape[2]*volume.shape[3]:
        volume_old = volume
        volume = np.zeros((volume.shape[0],volume.shape[1],FijiMeta['nslices']*FijiMeta['nchannels'],1),volume_old.dtype.name)
        for c in range(0,FijiMeta['nchannels']):
            volume[:,:,c:FijiMeta['nimages']:FijiMeta['nchannels'],0] = volume_old[:,:,:,c]
    
    # Write data to file
    if(path.find('.tif')==-1):
        path = path + '.tif'
    folderpath = '/'.join([folder for folder in path.split('/')[0:-1]])
    if(os.path.isdir(folderpath) == False):
        os.makedirs(folderpath) # create folder(s) if missing so far.
    outTiff = TIFF.open(path, mode='w')
    outTiff.SetField('ImageDescription', FijiMeta_IDb)
    outTiff.SetField('Photometric', FijiMeta['PhotometricInterpretation']) # libtiff_ctypes.py has different nomenclature
    outTiff.SetField('PlanarConfig', FijiMeta['PlanarConfiguration']) # libtiff_ctypes.py has different nomenclature
    outTiff.SetField('Orientation', FijiMeta['Orientation'])
    outTiff.SetField('ImageLength', FijiMeta['ImageLength'])
    outTiff.SetField('ImageWidth', FijiMeta['ImageWidth'])
    volume = np.rollaxis(volume, 3, 0)
    for zInd in range(volume.shape[3]):
        outTiff.write_image(volume[:,:,:,zInd],compression=compression, write_rgb=False)
    
    outTiff.close()
    return None



def writeNifti(path,volume,compress=False):
    '''
    writeNifti(path,volume)
    
    Takes a Numpy volume, converts it to the Nifti1 file format, and saves it to file under
    the specified path. 
    '''
    if(path.find('.nii')==-1 and compress==False):
        path = path + '.nii'
    if(path.find('.nii.gz')==-1 and compress==True):
        path = path + '.nii.gz'
    folderpath = '/'.join([folder for folder in path.split('/')[0:-1]])
    if(os.path.isdir(folderpath) == False):
        os.makedirs(folderpath) # create folder(s) if missing so far.
    # Save volume with adjusted orientation
    # --> Swap X and Y axis to go from (y,x,z) to (x,y,z)
    # --> Show in RAI orientation (x: right-to-left, y: anterior-to-posterior, z: inferior-to-superior)
    affmat = np.eye(4)
    affmat[0,0] = affmat[1,1] = -1
    NiftiObject = nib.Nifti1Image(np.swapaxes(volume,0,1), affine=affmat)
    nib.save(NiftiObject,path)


def readNifti(path,reorient=None):
    '''
    volume = readNifti(path)
    
    Reads in the NiftiObject saved under path and returns a Numpy volume.
    This function can also read in .img files (ANALYZE format).
    '''
    if(path.find('.nii')==-1 and path.find('.img')==-1):
        path = path + '.nii'
    print(path)
    if(os.path.isfile(path)):    
        NiftiObject = nib.load(path)
    elif(os.path.isfile(path + '.gz')):
        NiftiObject = nib.load(path + '.gz')
    else:
        raise Exception("No file found at: "+path)
    # Load volume and adjust orientation from (x,y,z) to (y,x,z)
    volume = np.swapaxes(NiftiObject.dataobj,0,1)
    if(reorient=='uCT_Rosenhain' and path.find('.img')):
        # Only perform this when reading in raw .img files
        # from the Rosenhain et al. (2018) dataset
        #    y = from back to belly
        #    x = from left to right
        #    z = from toe to head
        volume = np.swapaxes(volume,0,2) # swap y with z
        volume = np.flip(volume,0) # head  should by at y=0
        volume = np.flip(volume,2) # belly should by at x=0
    return volume



























