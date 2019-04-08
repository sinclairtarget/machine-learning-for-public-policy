import csv
from census import Census

TOTAL_POP = 'B01003_001E'
BLACK_POP = 'B02001_003E'
HISPANIC_POP = 'B03003_003E'
MEDIAN_AGE = 'B01002_001E'
HOUSEHOLD_INC = 'B19013_001E'

def rename_vars(tract):
    tract['total_pop'] = tract.pop(TOTAL_POP)
    tract['black_pop'] = tract.pop(BLACK_POP)
    tract['median_age'] = tract.pop(MEDIAN_AGE)
    tract['hispanic_pop'] = tract.pop(HISPANIC_POP)
    tract['household_inc'] = tract.pop(HOUSEHOLD_INC)

query_vars = (
    'NAME',
    TOTAL_POP,
    BLACK_POP,
    MEDIAN_AGE,
    HISPANIC_POP,
    HOUSEHOLD_INC
)

c = Census('35c615c69742eabcfe631cf4c29ff3fed90939cc')
resp = c.acs5.state_county_tract(query_vars, '17', '031', Census.ALL)

with open('data/census2018.csv', mode='w') as f:
    fieldnames = [
        'tract',
        'total_pop',
        'black_pop',
        'median_age',
        'hispanic_pop',
        'household_inc'
    ]

    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')

    writer.writeheader()

    for tract in resp:
        rename_vars(tract)
        writer.writerow(tract)
