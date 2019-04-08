def task_fetch_crimes():
    """Fetches Chicago crime data for 2017 and 2018."""
    return {
        'file_dep': ['fetch_crimes.py'],
        'targets': ['data/crimes2017.csv', 'data/crimes2018.csv'],
        'actions': ['pipenv run python %(dependencies)s']
    }

def task_spatial_join():
    """Runs a spatial join to attach tract IDs to crimes."""
    return {
        'file_dep': ['spatial_join.py'],
        'targets': ['data/crimes2018_geo.csv'],
        'actions': ['pipenv run python %(dependencies)s']
    }

def task_generate_report():
    """Generates PDF report from RMarkdown source."""
    return {
        'file_dep': ['crime_analysis.rmd',
                     'data/crimes2017.csv',
                     'data/crimes2018.csv',
                     'data/crimes2018_geo.csv'],
        'targets': ['crime_analysis.pdf'],
        'actions': ['pipenv run Rscript -e \'rmarkdown::render("crime_analysis.rmd")\'']
    }
