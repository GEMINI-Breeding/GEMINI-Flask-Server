# reprojection.py

import geopandas as gpd
import sys

def reproject(input_file, output_file):
    # Load your geojson data
    gdf = gpd.read_file(input_file)

    # Set the current CRS of the GeoDataFrame
    gdf.set_crs(epsg=32610, inplace=True)

    # Reproject to WGS 84 (EPSG:4326)
    gdf = gdf.to_crs(epsg=4326)

    # Write the reprojected data back to a new GeoJSON file
    gdf.to_file(output_file, driver='GeoJSON')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python reprojection.py <input_file> <output_file>")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        reproject(input_file, output_file)
