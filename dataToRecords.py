
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import os
import gc
import re
import time
import random
import logging
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import rasterio
from glob import glob
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Conv2DTranspose, 
                                     ZeroPadding2D, SeparableConv2D, Add, Dense, BatchNormalization, ReLU, 
                                     GlobalAvgPool2D, Reshape, Lambda, LSTM, concatenate, Activation, Dropout, 
                                     LeakyReLU)
from tensorflow.keras.metrics import Recall, Precision, categorical_accuracy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import EfficientNetV2S, EfficientNetV2B0, Xception


patchsize = 256

path = ""
outputpath = ""


@tf.function
def random_transform(dataset):
    x = tf.random.uniform(())

    if x < 0.10:
        dataset = tf.image.flip_left_right(dataset)
    elif tf.math.logical_and(x >= 0.10, x < 0.20):
        dataset = tf.image.flip_up_down(dataset)
    elif tf.math.logical_and(x >= 0.20, x < 0.30):
        dataset = tf.image.flip_left_right(tf.image.flip_up_down(dataset))
    elif tf.math.logical_and(x >= 0.30, x < 0.40):
        dataset = tf.image.rot90(dataset, k=1)
    elif tf.math.logical_and(x >= 0.40, x < 0.50):
        dataset = tf.image.rot90(dataset, k=2)
    elif tf.math.logical_and(x >= 0.50, x < 0.60):
        dataset = tf.image.rot90(dataset, k=3)
    elif tf.math.logical_and(x >= 0.60, x < 0.70):
        dataset = tf.image.flip_left_right(tf.image.rot90(dataset, k=2))
    else:
        pass
    return dataset


@tf.function
def flip_inputs_up_down(inputs):
    return tf.image.flip_up_down(inputs)


@tf.function
def flip_inputs_left_right(inputs):
    return tf.image.flip_left_right(inputs)


@tf.function
def transpose_inputs(inputs):
    flip_up_down = tf.image.flip_up_down(inputs)
    transpose = tf.image.flip_left_right(flip_up_down)
    return transpose


@tf.function
def rotate_inputs_90(inputs):
    return tf.image.rot90(inputs, k=1)


@tf.function
def rotate_inputs_180(inputs):
    return tf.image.rot90(inputs, k=2)


@tf.function
def rotate_inputs_270(inputs):
    return tf.image.rot90(inputs, k=3)



output_signature = (
    {
        "input_image1": tf.TensorSpec(shape=(None, None, 4), dtype=tf.float32),
        "input_image2": tf.TensorSpec(shape=(None, None, 4), dtype=tf.float32),
        "input_image3": tf.TensorSpec(shape=(None, None, 6), dtype=tf.float32),
    },
    tf.TensorSpec(shape=(None, None, 2), dtype=tf.float32),
)

# Function to generate a new random number and check for the file existence
def get_unique_filename(identifier,dirs,name):
    while True:
        nr = str(random.randint(0, 5)).zfill(2)
        rgbn_file = os.path.join(dirs, f"{name}{identifier}_{nr}.tif")
        # Check if the file exists
        if os.path.exists(rgbn_file):
            # If the file does not exist, break the loop and return the file name
            return rgbn_file

def get_unique_filename_month(identifier, dirs, name):
    months = ["jan", "feb", "mar", "apr", "jun", "oct", "nov", "dec"]

    while True:
        nr = random.randint(0, len(months) - 1)
        monthName = months[nr]
        planet_file = os.path.join(dirs, f"{name}_{identifier}_{monthName}.tif")

        # Check if the file does not exist
        if os.path.exists(planet_file):
            # If the file does not exist, return the file name
            return planet_file

def image_pair_generatorv1(identifiers,nr):
    def has_cashew(patch):
        return np.sum(patch == 1) > 256*256 * 0.05

    def clip_image(img,clip_pixels):
        return img[clip_pixels:-clip_pixels, clip_pixels:-clip_pixels, :]

    def clip_image_class(img,clip_pixels):
        return img[clip_pixels:-clip_pixels, clip_pixels:-clip_pixels]
    cashew_count = 0
    non_cashew_count = 0
    total_patches = nr
    max_imbalance=20

    
    for identifier in identifiers:
                
        dirs = os.path.join(path,"Labels")
        class_file = os.path.join(dirs, f"class_{identifier}.tif")
		
        nr = str(random.randint(0, 5)).zfill(2)
        dirs = os.path.join(path,"RGBN")
        rgbn_file = get_unique_filename(identifier,dirs,"rgbn_")
		
        dirs = os.path.join(path,"planet")
        name = "planet"
        planet_file = get_unique_filename_month(identifier, dirs, name)

        nr = str(random.randint(0, 5)).zfill(2)
        dirs = os.path.join(path,"OtherBands")
        other_file = get_unique_filename(identifier,dirs,"other_")
        
        nr = str(random.randint(0, 5)).zfill(2)
        dirs = os.path.join(path,"Landsat8")
        landsat_file = get_unique_filename(identifier,dirs,"l8_") 
        
        dirs = os.path.join(path,"Sentinel1")
        s1_file = os.path.join(dirs, f"s1_{identifier}.tif")

        filename = identifier

        with rasterio.open(class_file) as src:
            class_image = src.read(1).astype(np.float32)

        with rasterio.open(rgbn_file) as src:
            sat_image = src.read().astype(np.float32)

        with rasterio.open(planet_file) as src:
            planet_image = src.read().astype(np.float32)

        with rasterio.open(other_file) as src:
            other_image = src.read().astype(np.float32)

        with rasterio.open(landsat_file) as src:
            landsat_image = src.read().astype(np.float32)

        with rasterio.open(s1_file) as src:
            s1_image = src.read().astype(np.float32)

        sat_image = np.transpose(sat_image, (1, 2, 0)) #/ 10000 
        planet_image = np.transpose(planet_image, (1, 2, 0)) #/ 10000
        other_image = np.transpose(other_image, (1, 2, 0)) / 10000 
        s1_image = np.transpose(s1_image, (1, 2, 0)) / 10000 
        landsat_image = np.transpose(landsat_image, (1, 2, 0)) / 10000 

        sat_image[:, :, :3] /= 1000.0
        sat_image[:, :, 3] /= 10000.0
        
        planet_image[:, :, :3] /= 1000.0
        planet_image[:, :, 3] /= 10000.0
        
        sat_image = clip_image(sat_image,8)               # 10
        planet_image = clip_image(planet_image,16)        # 5
        class_image = clip_image_class(class_image,16)    # 5
        s1_image = clip_image(s1_image,8)              # 10 
        other_image = clip_image(other_image,4)           # 20
        landsat_image = clip_image(landsat_image,2)       # 40 


        cashew = class_image #np.where(class_image == 2, 1, 0)
        counter = 0

        for i in range(0, total_patches, 1):
                attempts = 0
                counter+=1
                sat_patch, planet_patch, other_patch, class_patch,s1_patch,landsat_patch =  random_patch_pair(sat_image, cashew, planet_image, other_image, landsat_image, s1_image,counter)
 
                contains_cashew = has_cashew(class_patch)

                imbalance = abs(cashew_count - non_cashew_count)
                yield sat_patch, planet_patch, other_patch, class_patch,s1_patch,landsat_patch
                    
        
        del sat_image, planet_image, other_image,s1_patch,landsat_patch
        gc.collect()




def random_patch_pair(sat_image, class_image, planet_image, other_image, landsat_image, s1_image,iteration, patch_size=(patchsize, patchsize)):
    """Randomly sample a patch from both an image and its corresponding class image."""
    
    # Set the patch size for each image based on its dimensions
    patch_size_rgbn = (patch_size[0] // 2, patch_size[1] // 2)
    patch_size_other = (patch_size[0] // 4, patch_size[1] // 4)
    patch_size_landsat = (patch_size[0] // 8, patch_size[1] // 8)
    
    current_time = int(time.time())  # Get the current time in seconds since the epoch
    np.random.seed(current_time+iteration)

    # Choose the top-left corner of the patch randomly
    start_i = np.random.randint(0, planet_image.shape[0] - patch_size[0] + 1)
    start_j = np.random.randint(0, planet_image.shape[1] - patch_size[1] + 1)

    # Slice out the patches, adjusting start and size for each image
    planet_patch = planet_image[start_i : start_i + patch_size[0], start_j : start_j + patch_size[1], :]
    #planet_patch = planet_image[start_i//2 : start_i//2 + patch_size_planet[0], start_j//2 : start_j//2 + patch_size_planet[1], :]
    class_patch = class_image[start_i : start_i + patch_size[0], start_j : start_j + patch_size[1]]
    
    sat_patch = sat_image[start_i //2: start_i//2 + patch_size_rgbn[0], start_j//2 : start_j//2 + patch_size_rgbn[1], :]   
    s1_patch = s1_image[start_i //2: start_i//2 + patch_size_rgbn[0], start_j//2 : start_j//2 + patch_size_rgbn[1], :]   
    

    other_patch = other_image[start_i//4 : start_i//4 + patch_size_other[0], start_j//4 : start_j//4 + patch_size_other[1], :]
    
    landsat_patch = landsat_image[start_i//8 : start_i//8 + patch_size_landsat[0], start_j//8 : start_j//8 + patch_size_landsat[1], :]



    # Apply random flip (horizontal and/or vertical)
    if np.random.rand() < 0.3: # 20% chance
        sat_patch = np.fliplr(sat_patch)
        class_patch = np.fliplr(class_patch)
        planet_patch = np.fliplr(planet_patch)
        other_patch = np.fliplr(other_patch)
        s1_patch = np.fliplr(s1_patch)
        landsat_patch = np.fliplr(landsat_patch)
     

    if np.random.rand() < 0.3: # 20% chance
        sat_patch = np.flipud(sat_patch)
        class_patch = np.flipud(class_patch)
        planet_patch = np.flipud(planet_patch)
        other_patch = np.flipud(other_patch)
        s1_patch = np.flipud(s1_patch)
        landsat_patch = np.flipud(landsat_patch)

    
    # Apply random rotation (0, 90, 180, or 270 degrees)
    if np.random.rand() < 0.3: 
        num_rotations = np.random.randint(4)
        sat_patch = np.rot90(sat_patch, num_rotations)
        class_patch = np.rot90(class_patch, num_rotations)
        planet_patch = np.rot90(planet_patch, num_rotations)
        other_patch = np.rot90(other_patch, num_rotations)
        s1_patch = np.rot90(s1_patch, num_rotations)
        landsat_patch = np.rot90(landsat_patch, num_rotations)
        #embed_patch = np.rot90(embed_patch, num_rotations)
    
    class_patch = np.expand_dims(class_patch, axis=-1)
    inverse_class_patch = 1 - class_patch
    final_class_patch = np.stack([class_patch, inverse_class_patch], axis=-1).astype(np.float32)
    return sat_patch, planet_patch, other_patch, class_patch, s1_patch,landsat_patch #final_class_patch #, embed_patch


    
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_patches_to_tfrecord(patches, filename):
    options = tf.io.TFRecordOptions(compression_type='GZIP')
    
    with tf.io.TFRecordWriter(filename, options=options) as writer:
        for sat_patch, planet_patch, other_patch, class_patch,s1_patch,landsat_patch in patches:
            
            # Convert arrays to bytes
            planet_patch_bytes = planet_patch.astype(np.float32).tobytes()
            sat_patch_bytes = sat_patch.astype(np.float32).tobytes()
            other_patch_bytes = other_patch.astype(np.float32).tobytes()
            class_patch_bytes = class_patch.astype(np.float32).tobytes()
            s1_patch_bytes = s1_patch.astype(np.float32).tobytes()
            landsat_patch_bytes = landsat_patch.astype(np.float32).tobytes()

            # Create a dictionary mapping the feature name to the tf.train.Feature
            feature = {
                'input_image1': _bytes_feature(planet_patch_bytes),
                'input_image2': _bytes_feature(sat_patch_bytes),
                'input_image3': _bytes_feature(other_patch_bytes),
                'input_image4': _bytes_feature(s1_patch_bytes),
                'input_image5': _bytes_feature(landsat_patch_bytes),
                'class_patch': _bytes_feature(class_patch_bytes),
            }
            # Create a Features message using tf.train.Example.
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example_proto.SerializeToString())



filenames= ["BMC_01", "BMC_02", "BMC_03", "BMC_05", "BMC_06", "BMC_07", "BMC_08", "BMC_09", "BMC_11",
    "BTB_01", "BTB_02", "BTB_03", "BTB_04", "BTB_05", "BTB_06", "BTB_07", "BTB_09", "BTB_10", "BTB_11", "BTB_12", "BTB_13",
    "KK_01", "KK_02", "KK_04", "KK_05", "KK_06", "KK_07", "KK_08", "KK_09", "KK_10",
    "KPC_01", "KPC_02", "KPC_03", "KPC_04", "KPC_05", "KPC_06", "KPC_07", "KPC_08", "KPC_09", "KPC_10",
    "KPCh_01", "KPCh_05", "KPCh_06", "KPCh_07", "KPCh_09", "KPCh_10",
    "KPT_01", "KPT_02", "KPT_03", "KPT_04", "KPT_05", "KPT_06", "KPT_07", "KPT_08", "KPT_09", "KPT_10", "KPT_11", "KPT_12", "KPT_13", "KPT_14", "KPT_16",
    "KP_01", "KP_08", "KP_09", "KP_11", "KP_12",
    "KPs_01", "KPs_02", "KPs_03", "KPs_04", "KPs_05", "KPs_06", "KPs_07", "KPs_08", "KPs_09", "KPs_10",
    "KT_01", "KT_02", "KT_04", "KT_05", "KT_06", "KT_07", "KT_08", "KT_09", "KT_10", "KT_11", "KT_12",
    "MDK_01", "MDK_02", "MDK_03", "MDK_04", "MDK_05", "MDK_06", "MDK_07", "MDK_08", "MDK_09",
    "OMC_01", "OMC_02", "OMC_03", "OMC_04", "OMC_05", "OMC_06", "OMC_07", "OMC_09", "OMC_10",
    "PL_01", "PL_02", "PL_03", "PL_05", "PL_06", "PL_08",
    "PSN_03", "PSN_05", "PSN_06", "PSN_08",
    "PS_02", "PS_03", "PS_04", "PS_05", "PS_06", "PS_07", "PS_08",
    "PVH_01", "PVH_02", "PVH_03", "PVH_04", "PVH_05", "PVH_06", "PVH_07", "PVH_08", "PVH_09", "PVH_10", "PVH_11", "PVH_12",
    "RTK_01", "RTK_02", "RTK_03", "RTK_04", "RTK_05", "RTK_06", "RTK_07", "RTK_08", "RTK_09", "RTK_10",
    "SR_01", "SR_02", "SR_03", "SR_04", "SR_05", "SR_06", "SR_07", "SR_08", "SR_09", "SR_10", "SR_11",
    "ST_01", "ST_02", "ST_03", "ST_04", "ST_05", "ST_06", "ST_07", "ST_08", "ST_09", "ST_10", "ST_11",
    "TbK_01", "TbK_02", "TbK_03", "TbK_04", "TbK_05", "TbK_06", "TbK_07", "TbK_08", "TbK_09", "TbK_10"]

for i in range(0,10,1):
    
    for identifier in filenames:
        try:
            print(identifier)
            nr = str(i).zfill(3)
            filename = f'{outputpath}/patch_{identifier}{nr}.tfrecord.gz'
            # Check if the file already exists
            if not os.path.exists(filename):
                patches = list(image_pair_generatorv1([identifier], 5))
                # Use the identifier to name the TFRecord for uniqueness
                write_patches_to_tfrecord(patches, filename)
            else:
                print(f"File {filename} already exists. Skipping...")
        except:
            print("skipping")
            pass
