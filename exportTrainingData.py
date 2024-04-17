from __future__ import annotations

import pandas as pd
import numpy as np
import warnings
import random
import time
import io
import os
import sys
import datetime
import json
import requests
from tqdm import tqdm
import concurrent.futures

from pyproj import Transformer
from PIL import Image
from sklearn import model_selection
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from osgeo import gdal
from pycocotools import mask
from skimage import measure
import cv2

import ee
from google.api_core import exceptions, retry
import google.auth
from numpy.lib.recfunctions import structured_to_unstructured

ee.Initialize()

initial_dict = ee.Dictionary({
    'Acacia': 0,
    'Annual Crop': 0,
    'Avocado': 0,
    'Bamboo': 0,
    'Banana': 1,
    'Bare land': 2,
    'Built up': 0,
    'Canal': 0,
    'Cashew': 3,
    'Cassava': 4,
    'Coconut': 5,
    'Crop land': 0,
    'Date palm': 0,
    'Dragon fruit': 0,
    'Durian': 6,
    'Eucalyptus': 0,
    'Forest Cover': 7,
    'Grassland': 8,
    'Guava': 0,
    'House': 9,
    'Jackfruit': 0,
    'Jujube': 0,
    'Lemon': 0,
    'Longan': 10,
    'Maize': 11,
    'Mango': 12,
    'Mangosteen': 0,
    'Mix Crops': 0,
    'Oil palm': 0,
    'Orange': 13,
    'Ornamental plant': 0,
    'Other': 14,
    'Pagoda': 0,
    'Papaya': 0,
    'Pepper': 15,
    'Rambutan': 0,
    'Resort': 0,
    'Rice': 16,
    'Road': 17,
    'Rubber': 18,
    'Shrubland': 19,
    'Solar farm': 0,
    'Sugarcane': 20,
    'Sweet potato': 21,
    'Sweetsop': 0,
    'Swine farm': 0,
    'Tree plantation': 0,
    'Vegetables': 22,
    'Village': 23,
    'Water Body': 24
})

areas = ["BMC_01", "BMC_02", "BMC_03", "BMC_05", "BMC_06", "BMC_07", "BMC_08", "BMC_09", "BMC_11",
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


# Create a transformer
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)

# Set the seed for the random number generator
np.random.seed(int(time.time()))

# Define a custom error handler function that does nothing
def handleError(err_class, err_num, err_msg):
    pass


def ee_init() -> None:
        """Authenticate and initialize Earth Engine with the default credentials."""
        # Use the Earth Engine High Volume endpoint.
        #   https://developers.google.com/earth-engine/cloud/highvolume
        credentials, project = google.auth.default(
            scopes=[
                "https://www.googleapis.com/auth/cloud-platform",
                "https://www.googleapis.com/auth/earthengine",
            ]
        )
        ee.Initialize(
            credentials.with_quota_project(None),
            project=project,
            opt_url="https://earthengine-highvolume.googleapis.com",
        )


@retry.Retry(deadline=10 * 60)  # seconds
def get_patch(
        image: ee.Image, lonlat: tuple[float, float], patch_size: int, scale: int
    ) -> np.ndarray:
        """Fetches a patch of pixels from Earth Engine.
        It retries if we get error "429: Too Many Requests".
        Args:
            image: Image to get the patch from.
            lonlat: A (longitude, latitude) pair for the point of interest.
            patch_size: Size in pixels of the surrounding square patch.
            scale: Number of meters per pixel.
        Raises:
            requests.exceptions.RequestException
        Returns: The requested patch of pixels as a NumPy array with shape (width, height, bands).
        """


        point = ee.Geometry.Point(lonlat)
        url = image.getDownloadURL(
            {
                "region": point.buffer(scale * patch_size / 2, 1).bounds(1),
                "dimensions": [patch_size, patch_size],
                "format": "NPY",
            }
        )

        # If we get "429: Too Many Requests" errors, it's safe to retry the request.
        # The Retry library only works with `google.api_core` exceptions.
        response = requests.get(url)
        if response.status_code == 429:
            raise exceptions.TooManyRequests(response.text)

        # Still raise any other exceptions to make sure we got valid data.
        response.raise_for_status()
        return np.load(io.BytesIO(response.content), allow_pickle=True), point.buffer(scale * patch_size / 2, 1).bounds(1)

def writeOutput(raster, out_file, patch_size, coords):
    """
    Create a new GeoTIFF file with the given filename and writes the given data into it.

    Args:
        counter (int): A counter to create a unique filename for each output file.
        out_file (str): The output filename to be created for each patch.
        patch_size (int): The size of the patch to be written to the output file.
        overlap_size (int): The size of the overlap between the patches.

    Returns:
        None
    """

    xmin = coords[0][0] #+ (overlap_size * pixel_size)
    xmax = coords[1][0] #- (overlap_size * pixel_size)
    ymin = coords[0][1] #+ (overlap_size * pixel_size)
    ymax = coords[2][1] #- (overlap_size * pixel_size)

    coords = [xmin,ymin,xmax,ymax]

    # Create a new GDAL driver for GeoTIFF
    driver = gdal.GetDriverByName("GTiff")

    l = raster.shape[2] 
    out_raster = driver.Create(out_file, patch_size, patch_size, l, gdal.GDT_Int16)

    # Set the spatial reference system (optional)
    out_raster.SetProjection("EPSG:4326")


    # Set the extent of the file
    out_raster.SetGeoTransform((xmin, (xmax - xmin) / (patch_size ), 0, ymax, 0, -(ymax - ymin) / (patch_size)))

    compress = "LZW"
    options = ["COMPRESS=" + compress]


    #out_band = out_raster.GetRasterBand(1)
    layer = raster
    #out_band.WriteArray(layer[:,:,0])
	
    for i in range(0,l,1):
        out_band = out_raster.GetRasterBand(i+1)
        out_band.WriteArray(layer[:,:,i])


def getImage(month, geom, year, item):
    """
    Retrieve composite images for a given month, geographic area, and year from the Google Earth Engine platform.

    Args:
        month (str): The month to retrieve the image for. Should be a three-letter abbreviation (e.g., 'jan', 'feb', etc.).
        geom (ee.Geometry): The geographical area for which the image is retrieved.
        year (int): The year to retrieve the image for.
        item (int): Index of the image to retrieve from the filtered ImageCollection.

    Returns:
        tuple: A tuple containing the RGBN composite, other bands composite, PlanetScope image,
               Landsat 8 image, and Sentinel-1 composite image.
    """

    # Mapping month abbreviations to Planet NICFI image IDs
    month_to_image_id = {
        "jan": "2022-01_mosaic",
        "feb": "2022-02_mosaic",
        "mar": "2022-03_mosaic",
        "apr": "2022-04_mosaic",
        "may": "2022-05_mosaic",
        "jun": "2022-06_mosaic",
        "oct": "2022-10_mosaic",
        "nov": "2022-11_mosaic",
        "dec": "2022-12_mosaic"
    }

    # Construct the PlanetScope image ID based on the month
    if month in month_to_image_id:
        planet = ee.Image(f"projects/planet-nicfi/assets/basemaps/asia/planet_medres_normalized_analytic_{month_to_image_id[month]}")

    # Define the time range for the satellite data
    startDate = ee.Date.fromYMD(year, 1, 1)
    endDate = ee.Date.fromYMD(year, 12, 31)
    
    # Retrieve Sentinel-2, Landsat 8, and Sentinel-1 images
    s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(geom).filterDate(startDate, endDate).sort("CLOUDY_PIXEL_PERCENTAGE")
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(geom).filterDate(startDate, endDate).sort("CLOUD_COVER")
    s1 = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(geom).filterDate(startDate, endDate).map(add_ratio)
   
    # Extract specific images based on the item index
    s2image = ee.Image(s2.toList(10).get(item))
    l8image = ee.Image(l8.toList(10).get(item)).select(["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]).multiply(0.0000275).add(-0.2)

    # Create RGBN and other band composites from Sentinel-2
    rgbn = s2image.select(["B2", "B3", "B4", "B8"])
    other = s2image.select(["B5", "B6", "B7", "B8A", "B11", "B12"])
    
    # Calculate median and standard deviation for Sentinel-1 images
    s1Median = s1.median()
    s1STD = s1.reduce(ee.Reducer.stdDev())
    s1Image = s1Median.addBands(s1STD)

    return rgbn, other, planet, l8image, s1Image

def add_ratio(img):
    """
    Add a ratio band to an input image.

    Args:
        img (ee.Image): The input image to add the ratio band to.

    Returns:
        ee.Image: The input image with a new ratio band.
    """
    geom = img.geometry()
    vv = to_natural(img.select(['VV'])).rename(['VV'])
    vh = to_natural(img.select(['VH'])).rename(['VH'])
    vv = vv.clamp(0, 1)  # Clamping VV values between 0 and 1
    vh = vh.clamp(0, 1)  # Clamping VH values between 0 and 
    return ee.Image(ee.Image.cat(vv, vh).copyProperties(img, ['system:time_start'])).clip(geom).copyProperties(img)

def erode_geometry(image):
    """
    Erode the geometry of an input image.

    Args:
        image (ee.Image): The input image to erode.

    Returns:
        ee.Image: The input image with eroded geometry.
    """
    return image.clip(image.geometry().buffer(-1000))

def to_natural(img):
    """
    Convert an input image from dB to natural.

    Args:
        img (ee.Image): The input image to convert.
    Returns:
        ee.Image: The input image in natural scale.
    """
    return ee.Image(10.0).pow(img.select(0).divide(10.0))

def toNatural(img):
    """Function to convert from dB"""
    return ee.Image(10.0).pow(img.select(0).divide(10.0))

def toDB(img):
    """Function to convert to dB"""
    return ee.Image(img).log10().multiply(10.0)

def ensure_directory_exists(file_path):
    """
    Ensure the directory for the given file path exists. If not, create it.

    Args:
        file_path (str): The full path to the file for which the directory needs to be ensured.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def RefinedLee(img):
    """The RL speckle filter
    img must be in natural units, i.e. not in dB!
    Set up 3x3 kernels"""
    bandNames = img.bandNames()
    img = toNatural(img)
    
    weights3 = ee.List.repeat(ee.List.repeat(1,3),3)
    kernel3 = ee.Kernel.fixed(3,3, weights3, 1, 1, False)

    mean3 = img.reduceNeighborhood(ee.Reducer.mean(), kernel3)
    variance3 = img.reduceNeighborhood(ee.Reducer.variance(), kernel3)

    # Use a sample of the 3x3 windows inside a 7x7 windows to determine gradients and directions
    sample_weights = ee.List([[0,0,0,0,0,0,0], [0,1,0,1,0,1,0],[0,0,0,0,0,0,0], [0,1,0,1,0,1,0], [0,0,0,0,0,0,0], [0,1,0,1,0,1,0],[0,0,0,0,0,0,0]])

    sample_kernel = ee.Kernel.fixed(7,7, sample_weights, 3,3, False)

    # Calculate mean and variance for the sampled windows and store as 9 bands
    sample_mean = mean3.neighborhoodToBands(sample_kernel)
    sample_var = variance3.neighborhoodToBands(sample_kernel)

    # Determine the 4 gradients for the sampled windows
    gradients = sample_mean.select(1).subtract(sample_mean.select(7)).abs()
    gradients = gradients.addBands(sample_mean.select(6).subtract(sample_mean.select(2)).abs())
    gradients = gradients.addBands(sample_mean.select(3).subtract(sample_mean.select(5)).abs())
    gradients = gradients.addBands(sample_mean.select(0).subtract(sample_mean.select(8)).abs())

    # And find the maximum gradient amongst gradient bands
    max_gradient = gradients.reduce(ee.Reducer.max())

    # Create a mask for band pixels that are the maximum gradient
    gradmask = gradients.eq(max_gradient)

    # duplicate gradmask bands: each gradient represents 2 directions
    gradmask = gradmask.addBands(gradmask)

    # Determine the 8 directions
    directions = sample_mean.select(1).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(7))).multiply(1)
    directions = directions.addBands(sample_mean.select(6).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(2))).multiply(2))
    directions = directions.addBands(sample_mean.select(3).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(5))).multiply(3))
    directions = directions.addBands(sample_mean.select(0).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(8))).multiply(4))
    # The next 4 are the not() of the previous 4
    directions = directions.addBands(directions.select(0).Not().multiply(5))
    directions = directions.addBands(directions.select(1).Not().multiply(6))
    directions = directions.addBands(directions.select(2).Not().multiply(7))
    directions = directions.addBands(directions.select(3).Not().multiply(8))

    # Mask all values that are not 1-8
    directions = directions.updateMask(gradmask)

    # "collapse" the stack into a singe band image (due to masking, each pixel has just one value (1-8) in it's directional band, and is otherwise masked)
    directions = directions.reduce(ee.Reducer.sum())

    sample_stats = sample_var.divide(sample_mean.multiply(sample_mean))

    # Calculate localNoiseVariance
    sigmaV = sample_stats.toArray().arraySort().arraySlice(0,0,5).arrayReduce(ee.Reducer.mean(), [0])

    # Set up the 7*7 kernels for directional statistics
    rect_weights = ee.List.repeat(ee.List.repeat(0,7),3).cat(ee.List.repeat(ee.List.repeat(1,7),4))

    diag_weights = ee.List([[1,0,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,1,0,0,0,0],
    [1,1,1,1,0,0,0], [1,1,1,1,1,0,0], [1,1,1,1,1,1,0], [1,1,1,1,1,1,1]])

    rect_kernel = ee.Kernel.fixed(7,7, rect_weights, 3, 3, False)
    diag_kernel = ee.Kernel.fixed(7,7, diag_weights, 3, 3, False)

    # Create stacks for mean and variance using the original kernels. Mask with relevant direction.
    dir_mean = img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel).updateMask(directions.eq(1))
    dir_var = img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel).updateMask(directions.eq(1))

    dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel).updateMask(directions.eq(2)))
    dir_var= dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel).updateMask(directions.eq(2)))

    # and add the bands for rotated kernels
    for i in range(1,4):
        dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)))
        dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)))
        dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)))
        dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)))
        
    # "collapse" the stack into a single band image (due to masking, each pixel has just one value in it's directional band, and is otherwise masked)
    dir_mean = dir_mean.reduce(ee.Reducer.sum())
    dir_var = dir_var.reduce(ee.Reducer.sum())

    # A finally generate the filtered value
    varX = dir_var.subtract(dir_mean.multiply(dir_mean).multiply(sigmaV)).divide(sigmaV.add(1.0))

    b = varX.divide(dir_var)

    result = dir_mean.add(b.multiply(img.subtract(dir_mean))).arrayFlatten([['sum']]) \
            .float()
            
    return ee.Image(toDB(result)).rename(bandNames).copyProperties(img)
    
def get_feature(feature_collection):
    """
    Adds a unique number to each feature in a feature collection and creates a dictionary of the lc_planet and unique_number properties.

    Args:
        feature_collection (ee.FeatureCollection): The feature collection to add unique numbers to.

    Returns:
        A tuple containing the unique numbers raster and a dictionary of lc_planet and unique_number lists.
    """
    # Define a function to add a unique number to each feature
    def add_unique_number(feature, list):
        index = ee.List(list).size()
        new_feature = ee.Feature(feature).set("unique_number", index.add(1))
        return ee.List(list).add(new_feature)

    # Iterate over the FeatureCollection to add unique numbers
    feature_list_with_unique_numbers = feature_collection.iterate(add_unique_number, ee.List([]))

    # Convert the list back to a FeatureCollection
    feature_collection_with_unique_numbers = ee.FeatureCollection(ee.List(feature_list_with_unique_numbers))

    # Get the lc_planet and unique_number properties from the feature collection
    lc_planet_list = feature_collection_with_unique_numbers.aggregate_array('lc_planet').getInfo()
    unique_number_list = feature_collection_with_unique_numbers.aggregate_array('unique_number').getInfo()
    classes = feature_collection_with_unique_numbers.aggregate_array('classification').getInfo()

    # Create a dictionary with the two lists
    lc_unique_dict = {
      'lc_planet': lc_planet_list,
      'unique_number': unique_number_list
    }

    # Reduce the FeatureCollection to an image
    unique_numbers_raster = feature_collection_with_unique_numbers.reduceToImage(["unique_number"], ee.Reducer.first())
    classification = feature_collection_with_unique_numbers.reduceToImage(["classification"], ee.Reducer.first())

    return unique_numbers_raster, classification, lc_unique_dict, classes
    

# Set parameters
month = "jan"
scale = 5
patch_size = 800  # Adjust patch size if necessary
year = 2022
item = 3
base_output_path = f""

# Specify the feature collection name and filter it by 'plot_id'
feature_collection_name = "RTK_01"
feature_collection_path = ""
ft = ee.FeatureCollection(feature_collection_path).filter(ee.Filter.eq('plot_id', feature_collection_name))

# Get the centroid of the feature collection
centroid = ft.geometry().centroid()

# Filter the feature collection by land cover keys
keys = initial_dict.keys()
ft = ft.filter(ee.Filter.inList("c_lu", keys))

# Function to remap class labels
def remap_classes(feature):
    old_class = feature.get('c_lu')
    new_class = initial_dict.get(old_class)
    return feature.set('classification', new_class)

# Apply the remap function to all features
ft = ft.map(remap_classes)

# Retrieve the size of the filtered feature collection
ft_size = ft.size().getInfo()

# Generate output name
output_name = f"{feature_collection_name}"
feature_count = int(ft.size().getInfo())

# Calculate bounding box and its coordinates
bounds = centroid.buffer((patch_size / 2) * scale).bounds()
coords = np.array(bounds.getInfo().get("coordinates"))[0]
xmin, ymin = coords[0][0], coords[0][1]  # Minimum x and y coordinates
xmax, ymax = coords[1][0], coords[2][1]  # Maximum x and y coordinates

# Calculate the central longitude and latitude
lon, lat = (xmin + xmax) / 2, (ymin + ymax) / 2
lonlat = (lon, lat)

FeatImage, classification, lc_unique_dict, classes = get_feature(ft)

# Process and export classification image
patch, bounds = get_patch(classification, lonlat, patch_size, scale)
classificationImg = structured_to_unstructured(patch)
label_output_path = os.path.join(base_output_path, "Labels", f"class_{feature_collection_name}.tif")
ensure_directory_exists(label_output_path)
print(f"Writing classification image to: {label_output_path}")
writeOutput(classificationImg, label_output_path, patch_size, coords)

# Retrieve images
rgbn, other, planet, l8, s1 = getImage(month, bounds, year, item)

# Process and export Sentinel-1 image
patch, bounds = get_patch(planet, lonlat, 800, 10)
planet = structured_to_unstructured(patch) 
planet_output_path = os.path.join(base_output_path, "planet", f"planet_{feature_collection_name}_{month}.tif")
ensure_directory_exists(planet_output_path)
print(f"Writing planet image to: {planet_output_path}")
writeOutput(planet, planet_output_path, 800, coords)


# Process and export Sentinel-1 image
patch, bounds = get_patch(s1, lonlat, 400, 10)
sentinel1 = structured_to_unstructured(patch) * 10000
s1_output_path = os.path.join(base_output_path, "Sentinel1", f"s1_{feature_collection_name}.tif")
ensure_directory_exists(s1_output_path)
print(f"Writing Sentinel-1 image to: {s1_output_path}")
writeOutput(sentinel1, s1_output_path, 400, coords)

# Process and export Landsat 8 image
patch, bounds = get_patch(l8, lonlat, 100, 40)
landsat = structured_to_unstructured(patch) * 10000
l8_output_path = os.path.join(base_output_path, "Landsat8", f"l8_{feature_collection_name}_{str(item).zfill(2)}.tif")
ensure_directory_exists(l8_output_path)
print(f"Writing Landsat 8 image to: {l8_output_path}")
writeOutput(landsat, l8_output_path, 100, coords)

# Process and export RGBN image from Sentinel-2
patch, geom = get_patch(rgbn, lonlat, 400, 10)
rgbn_img = structured_to_unstructured(patch)
rgbn_output_path = os.path.join(base_output_path, "RGBN", f"rgbn_{feature_collection_name}_{str(item).zfill(2)}.tif")
ensure_directory_exists(rgbn_output_path)
print(f"Writing RGBN image to: {rgbn_output_path}")
writeOutput(rgbn_img, rgbn_output_path, 400, coords)

# Process and export Other bands image from Sentinel-2
patch, geom = get_patch(other, lonlat, 200, 20)
other_bands = structured_to_unstructured(patch)
other_output_path = os.path.join(base_output_path, "OtherBands", f"other_{feature_collection_name}_{str(item).zfill(2)}.tif")
ensure_directory_exists(other_output_path)
print(f"Writing Other bands image to: {other_output_path}")
writeOutput(other_bands, other_output_path, 200, coords)







