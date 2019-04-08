def task_fetch_crimes():
    """Fetches Chicago crime data for 2017 and 2018."""
    return {
        'file_dep': ['fetch_crimes.py'],
        'targets': ['data/crimes2017.csv', 'data/crimes2018.csv'],
        'actions': ['pipenv run python %(dependencies)s']
    }

def task_spatial_join():
    """Runs a spatial join to attach tract IDs to 2018 crimes."""
    return {
        'file_dep': ['spatial_join.py',
                     'data/crimes2018.csv',
                     'data/census.shp'],
        'targets': ['data/crimes2018_geo.csv'],
        'actions': ['pipenv run python spatial_join.py']
    }

def task_fetch_census():
    """Fetches 2018 augmentation data from census API."""
    return {
        'file_dep': ['fetch_census.py'],
        'targets': ['data/census2018.csv'],
        'actions': ['pipenv run python %(dependencies)s']
    }

def task_augment():
    """Augments 2018 crime data with census data."""
    return {
        'file_dep': ['augment.py',
                     'data/crimes2018.csv',
                     'data/crimes2018_geo.csv',
                     'data/census2018.csv'],
        'targets': ['data/crimes2018_augmented.csv'],
        'actions': ['pipenv run python augment.py']
    }

def task_generate_report():
    """Generates PDF report from RMarkdown source."""
    return {
        'file_dep': ['crime_analysis.rmd',
                     'data/crimes2017.csv',
                     'data/crimes2018.csv',
                     'data/crimes2018_augmented.csv'],
        'targets': ['crime_analysis.pdf'],
        'actions': ['pipenv run Rscript -e \'rmarkdown::render("crime_analysis.rmd")\'']
    }
