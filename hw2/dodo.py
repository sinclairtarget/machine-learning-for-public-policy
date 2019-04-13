def task_clean_data():
    """Cleans data and generates features."""
    return {
        'file_dep': ['clean.py', 'data/credit-data.csv'],
        'targets': ['data/credit-data-cleaned.csv'],
        'actions': ['pipenv run python clean.py']
    }

def task_predict():
    """Uses cleaned data to generate predictions."""
    return {
        'file_dep': ['predict.py', 'data/credit-data-cleaned.csv'],
        'targets': ['data/credit-data-predicted.csv'],
        'actions': ['pipenv run python predict.py']
    }

def task_evaluate():
    """Uses predicted data to evaluate accuracy."""
    return {
        'file_dep': ['evaluate.py', 'data/credit-data-predicted.csv'],
        'actions': ['pipenv run python evaluate.py'],
        'verbosity': 2,
        'uptodate': [False]
    }
