#! /bin/bash

pipenv run Rscript -e 'rmarkdown::render("crime_analysis.rmd")'
