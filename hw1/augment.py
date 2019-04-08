# This scripts joins Chicago crime data with ACS data for demographic
# background context.

import pandas as pd
from pandas import Series, DataFrame

crimes = pd.read_csv('data/crimes2018.csv', dtype={'ID': object})
crimes_geo = pd.read_csv('data/crimes2018_geo.csv',
                         dtype={'ID': object, 'tractce10': object})
census = pd.read_csv('data/census2018.csv', dtype={'tract': object})

# Join crimes with census tract identifiers from spatial join
crimes = pd.merge(crimes, crimes_geo, how='left', on='ID')

# Get rid of crimes that we don't have tract info for
len_before = len(crimes)
crimes = crimes[pd.notnull(crimes['tractce10'])]
len_after = len(crimes)

print(f"Dropped {len_before - len_after} crimes that could not be matched"
       " with a census tract.")

# Join crimes with census data for tracts
augmented = pd.merge(crimes, census, how='inner', left_on='tractce10',
                     right_on='tract')
augmented.to_csv('data/crimes2018_augmented.csv', index=False)
