def task_generate_report():
    """Generates PDF report from RMarkdown source."""
    return {
        'file_dep': ['report.rmd'],
        'targets': ['report.pdf'],
        'actions': ['pipenv run Rscript -e \'rmarkdown::render("report.rmd")\'']
    }
