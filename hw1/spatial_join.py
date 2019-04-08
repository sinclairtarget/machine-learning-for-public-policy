# This script uses geopandas and a census tract shapefile to add census tract
# identifiers to each crime with lat/long coordinates.

import pandas as pd
from pandas import Series, DataFrame
from shapely.geometry import Point
import geopandas as gpd

def load_crime_coords(csv_filename):
    df = pd.read_csv(csv_filename)

    coords = df[df.Longitude.notnull()][['ID', 'Longitude', 'Latitude']]
    coords['Coordinates'] = list(zip(coords.Longitude, coords.Latitude))
    coords['Coordinates'] = coords['Coordinates'].apply(Point)

    return gpd.GeoDataFrame(coords, geometry='Coordinates')

def load_tracts(shp_filename):
    return gpd.read_file(shp_filename)

if __name__ == "__main__":
    crimes = load_crime_coords('data/crimes2018.csv')
    crimes.crs = {'init': 'epsg:4326'}

    tracts = load_tracts('data/census.shp')
    tracts = tracts[['name10', 'namelsad10', 'tractce10', 'geoid10', 'geometry']]

    joined = gpd.sjoin(crimes, tracts, how='inner', op='within')
    joined = joined[['ID', 'name10', 'tractce10', 'geoid10']]

    joined.to_csv('data/crimes2018_geo.csv', index=False)
