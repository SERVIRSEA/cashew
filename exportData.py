from PIL import Image
import cv2
import requests
from osgeo import osr, gdal
import ee
from google.api_core import exceptions, retry
import numpy as np
import io
from numpy.lib.recfunctions import structured_to_unstructured
from segment_anything import sam_model_registry,SamAutomaticMaskGenerator, SamPredictor
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin
from datetime import datetime
import os

import concurrent.futures
from concurrent.futures import ProcessPoolExecutor


ee.Initialize()

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

def rescale_rgb(arr, lower_bound=0, upper_bound=255, lower_percentile=5, upper_percentile=95):
    """
    Rescales the pixel values of an RGB image to the specified range.

    Args:
        arr (np.ndarray): The RGB image to rescale.
        lower_bound (int): The lower bound of the output range.
        upper_bound (int): The upper bound of the output range.
        lower_percentile (float): The percentile to use as the lower limit for clipping outliers.
        upper_percentile (float): The percentile to use as the upper limit for clipping outliers.

    Returns:
        The rescaled RGB image as a uint8 numpy array.
    """
    # Compute the percentiles for each channel
    lower_limit = np.percentile(arr, lower_percentile, axis=(0, 1))
    upper_limit = np.percentile(arr, upper_percentile, axis=(0, 1))

    # Clip the values to remove outliers
    clipped_arr = np.clip(arr, lower_limit, upper_limit)

    # Rescale the values to the desired range
    rescaled_arr = ((clipped_arr - lower_limit) / (upper_limit - lower_limit)) * (upper_bound - lower_bound) + lower_bound

    return rescaled_arr.astype(np.uint8)


def getImage(month,geom,year):
    """
    Retrieve an image from the Google Earth Engine platform by filename.

    Args:
        fname (str): The filename of the image to retrieve, in the format "S2A_MSIL2A_YYYYMMDDTHHMMSS_XXXX_RRRR_TTTTTT_YYYYMMDDTHHMMSS".

    Returns:
        ee.Image: The image selected from the Google Earth Engine platform.
    """

    if month == "jan":
        planet = ee.Image("projects/planet-nicfi/assets/basemaps/asia/planet_medres_normalized_analytic_2022-01_mosaic")
    if month == "feb":
        planet = ee.Image("projects/planet-nicfi/assets/basemaps/asia/planet_medres_normalized_analytic_2022-02_mosaic")
    if month == "mar":
        planet = ee.Image("projects/planet-nicfi/assets/basemaps/asia/planet_medres_normalized_analytic_2022-03_mosaic")
    if month == "apr":
        planet = ee.Image("projects/planet-nicfi/assets/basemaps/asia/planet_medres_normalized_analytic_2022-04_mosaic")
    if month == "may":
        planet = ee.Image("projects/planet-nicfi/assets/basemaps/asia/planet_medres_normalized_analytic_2022-05_mosaic")
    if month == "jun":
        planet = ee.Image("projects/planet-nicfi/assets/basemaps/asia/planet_medres_normalized_analytic_2022-06_mosaic")
    if month == "oct":
        planet = ee.Image("projects/planet-nicfi/assets/basemaps/asia/planet_medres_normalized_analytic_2022-10_mosaic")
    if month == "nov":
        planet = ee.Image("projects/planet-nicfi/assets/basemaps/asia/planet_medres_normalized_analytic_2022-11_mosaic")
    if month == "dec":
        planet = ee.Image("projects/planet-nicfi/assets/basemaps/asia/planet_medres_normalized_analytic_2022-12_mosaic")

    planet = planet.select(["R","G","B","N"])
    
    startDate = ee.Date.fromYMD(year,1,1)
    endDate = ee.Date.fromYMD(year+1,1,1)
    s2 =ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(geom).filterDate(startDate,endDate).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",100)).sort("CLOUDY_PIXEL_PERCENTAGE")
    
    image = ee.Image(s2.toList(10).get(0))
    timestamp = ee.Date(image.get('system:time_start'))
    
    image = ee.Image(s2.reduce(ee.Reducer.firstNonNull())).select(['B1_first', 'B2_first', 'B3_first', 'B4_first', 'B5_first', 'B6_first', 'B7_first', 'B8_first', 'B8A_first', 'B9_first', 'B11_first', 'B12_first'],['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'])

    rgbn = image.select(["B2","B3","B4","B8"])
    other = image.select(["B5","B6","B7","B8A","B11","B12"])
    	
    start_date = timestamp.advance(-90, 'day')  # 30 days before
    end_date = timestamp.advance(90, 'day')     # 30 days after		
    
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(geom).filterDate(startDate,endDate).sort("CLOUD_COVER")
    l8image = ee.Image(l8.first())
    l8image = l8image.select(["SR_B1","SR_B2","SR_B3","SR_B4","SR_B5","SR_B6","SR_B7"]).multiply(0.0000275).add(-0.2)
    
    s1 = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(geom).filterDate(startDate,endDate).map(add_ratio)
    s1Median = s1.median()
    s1STD = s1.reduce(ee.Reducer.stdDev())
    s1Image = s1Median.addBands(s1STD)

    return rgbn, other, planet, l8image, s1Image


year = 2023
m = 12
scale = 4.7
patch_size = 1024 + 256
pxsize = scale * (1.0  / 111320.0 ) * (patch_size /2)


def write_array_to_geotif(data, coords, timestamp_str, dirp):

    lower_left_x, lower_left_y = coords[0][0][0], coords[0][0][1]  
    upper_left_x, upper_left_y = coords[0][3][0], coords[0][3][1]  
    upper_right_x, upper_right_y = coords[0][2][0], coords[0][2][1]  
    lower_right_x, lower_right_y = coords[0][1][0], coords[0][1][1]  

    x_pixel_size = (upper_right_x - lower_left_x) / data.shape[1]
    y_pixel_size = (upper_left_y - lower_left_y) / data.shape[0]

    nrows, ncols = data.shape

    filename = f'{dirp}{timestamp_str}.tif'
    
    driver = gdal.GetDriverByName('GTiff')
    
    # Set compression option
    options = ['COMPRESS=DEFLATE']
    dataset = driver.Create(filename, ncols, nrows, 1, gdal.GDT_Int16, options)

    dataset.SetGeoTransform((lower_left_x, x_pixel_size, 0, upper_left_y, 0, -y_pixel_size))

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326) 
    dataset.SetProjection(srs.ExportToWkt())

    band = dataset.GetRasterBand(1)

    nodata_value = 0
    band.SetNoDataValue(nodata_value)

    band.WriteArray(data)

    dataset.FlushCache()
    dataset = None

def writeOutputRGBN(raster, out_file, patch_size, coords):
    xmin = xmax = coords[0][0][0]
    ymin = ymax = coords[0][0][1]

    for x, y in coords[0]:
        xmin = min(xmin, x)
        xmax = max(xmax, x)
        ymin = min(ymin, y)
        ymax = max(ymax, y)

    coords = [xmin, ymin, xmax, ymax]
    driver = gdal.GetDriverByName("GTiff")

    l = raster.shape[2]

    compress = "LZW"
    options = ["COMPRESS=" + compress]

    out_raster = driver.Create(out_file, patch_size, patch_size, l, gdal.GDT_Int16, options=options)
    out_raster.SetProjection("EPSG:4326")
    out_raster.SetGeoTransform((xmin, (xmax - xmin) / patch_size, 0, ymax, 0, -(ymax - ymin) / patch_size))

    layer = raster
    for i in range(0, l, 1):
        out_band = out_raster.GetRasterBand(i + 1)
        out_band.WriteArray(layer[:, :, i])

    out_raster = None



fc = ee.FeatureCollection('projects/servir-mekong/khProject/khGeom');

size = fc.size().getInfo()

fc_list = fc.toList(size)

# Function to check if a file exists
def file_exists(filename):
    return os.path.exists(filename)


output_directory = "/path/to"   
# Iterate over the list to access each feature.
def process_feature(i):
    print("processing item.. ", str(i))
    
    feature = ee.Feature(fc_list.get(i))
    geometry = feature.geometry()
        
    # Define output file paths
    rgbn_output_file = output_directory + "rgbn_" + str(i).zfill(4) + ".tif"
    other_output_file = output_directory + "other_" + str(i).zfill(4) + ".tif"
    planet_output_file = output_directory + "planet_" + str(i).zfill(4) + ".tif"
    sar_output_file = output_directory + "s1_" + str(i).zfill(4) + ".tif"

    # Skip processing if the output files already exist
    if (file_exists(sar_output_file)):
        print("Files already exist. Skipping.. ", i)
        return     
      
       
    # Use the coordinates() function to extract the coordinates of the feature.
    coords = ee.List(geometry.coordinates()).getInfo()
    
    lon = coords[0]
    lat = coords[1]
    lonlat = (lon,lat)
    
    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")        

    rgbn, other, planet,l8,s1 = getImage("jun",geometry,year)
        
        
    patch, geom = get_patch(rgbn, lonlat, patch_size//2, 10)
    rgbn = structured_to_unstructured(patch)

    patch, geom = get_patch(other, lonlat,patch_size//4, 20)
    other = structured_to_unstructured(patch)

    patch, geom = get_patch(l8, lonlat,patch_size//8, 40)
    landsat = structured_to_unstructured(patch) * 10000

    patch, geom = get_patch(s1, lonlat,patch_size//2, 10)
    sentinel1 = structured_to_unstructured(patch)* 10000 

    patch, geom = get_patch(planet, lonlat, patch_size, 5)
    planet = structured_to_unstructured(patch)

    coords = np.array(geom.getInfo()['coordinates'])
        
    writeOutputRGBN(rgbn, output_directory + "rgbnToa_" + str(i).zfill(4) +".tif", patch_size//2, coords)
    writeOutputRGBN(other, output_directory + "otherToa_" + str(i).zfill(4) +".tif", patch_size//4, coords)
    writeOutputRGBN(sentinel1, output_directory + "s1_" + str(i).zfill(4) +".tif", patch_size//2, coords)
    writeOutputRGBN(landsat, output_directory + "l8_" + str(i).zfill(4) +".tif", patch_size//8, coords)
    writeOutputRGBN(planet, output_directory + "planet_" + str(i).zfill(4) +".tif", patch_size, coords)


year = 2022
#process_feature(1)


# Execute process_feature in parallel
with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_feature, i) for i in range(0, 7300, 1)]

# Wait for all processes to complete
concurrent.futures.wait(futures)
print("All tasks completed.")

