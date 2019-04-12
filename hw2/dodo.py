def task_clean_data():
    """Cleans data and generates features."""
    return {
        'file_dep': ['clean.py', 'data/credit-data.csv'],
        'targets': ['data/credit-data-cleaned.csv'],
        'actions': ['pipenv run python clean.py']
    }
