DOIT_CONFIG = {
    'default_tasks': ['generate_analysis_html']
}

def task_generate_analysis_html():
    """Generates HTML Jupyter notebook file from notebook source."""
    return {
        'file_dep': ['analysis.ipynb'],
        'targets': ['analysis.html'],
        'actions': ['pipenv run jupyter nbconvert --execute --to html'
                    ' --ExecutePreprocessor.timeout=360 analysis.ipynb'],
        'clean': True
    }


def task_generate_analysis_notebook():
    """Generates Jupyter notebook from markdown source."""
    return {
        'file_dep': ['analysis.md'],
        'targets': ['analysis.ipynb'],
        'actions': ['pipenv run jupytext --to notebook analysis.md'],
        'clean': True                   # Remove all targets
    }


def task_generate_report():
    """Generates PDF report from markdown source."""
    return {
        'file_dep': ['report.md'],
        'targets': ['report.pdf'],
        'actions': ['pandoc -t latex -o report.pdf report.md'],
        'clean': True
    }
