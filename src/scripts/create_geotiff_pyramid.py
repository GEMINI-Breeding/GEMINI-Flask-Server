import argparse
from osgeo import gdal

def create_tiled_pyramid(input_path, output_path):
    # Open the original raster in read mode
    ds = gdal.Open(input_path, gdal.GA_ReadOnly)

    # Check the projection information
    projection = ds.GetProjection()
    print('Original Projection:', projection)

    # Reproject the raster to Web Mercator
    print('Reprojecting...')
    ds = gdal.Warp('', ds, format='MEM', dstSRS='EPSG:3857')
    projection = ds.GetProjection()
    print('New Projection:', projection)

    # Create a new raster in write mode with the same properties as the original raster
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.CreateCopy(output_path, ds)

    # Set the GeoTIFF options
    dst_ds.SetMetadata({'TILED': 'YES', 'BIGTIFF': 'YES', 'COMPRESS': 'LZW'})

    # Close the datasets to flush to disk
    ds = None
    dst_ds = None

    # Create the pyramid layers
    ds = gdal.Open(output_path, gdal.GA_Update)
    ds.BuildOverviews('AVERAGE', [2, 4, 8, 16, 32, 64, 128, 256])
    ds = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a tiled GeoTIFF with pyramid layers.')
    parser.add_argument('input', type=str, help='Path to the input raster file.')
    parser.add_argument('output', type=str, help='Path to the output tiled GeoTIFF file.')
    args = parser.parse_args()

    create_tiled_pyramid(args.input, args.output)
