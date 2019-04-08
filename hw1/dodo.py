def task_fetch_crimes():
    """Fetches Chicago crime data for 2017 and 2018."""
    return {
        'file_dep': ['fetch_crimes.py'],
        'targets': ['data/crimes2017.csv', 'data/crimes2018.csv'],
        'actions': ['pipenv run python %(dependencies)s']
    }
