DOIT_CONFIG = {
    'default_tasks': ['generate_analysis_html']
}

def task_render():
    """Generates HTML Jupyter notebook file from notebook source."""
    target = 'analysis.html'
    dep = 'analysis.ipynb'
    return {
        'file_dep': [dep],
        'targets': [target],
        'actions': [
            f"jupyter nbconvert --execute --to html {dep}"
        ],
        'clean': True
    }


def task_sync():
    """Generates Jupyter notebook from markdown source."""
    target = 'analysis.ipynb'
    dep = 'analysis.md'
    return {
        'file_dep': [dep],
        'targets': [target],
        'actions': [
            f"jupytext --to notebook {dep}",
            f"jupytext --set-format ipynb,md --sync {target}"
        ],
        'clean': True                   # Remove all targets
    }


def task_render_report():
    """Generates PDF report from markdown source."""
    target = 'report.pdf'
    dep = 'report.md'
    return {
        'file_dep': [dep],
        'targets': [target],
        'actions': [
            f"pandoc -t latex -o {target} {dep}"
        ],
        'clean': True
    }
