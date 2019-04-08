# Diagnostic HW
## Python Environment
Pipenv is used to handle Python dependencies.

## External Dependencies
R and RMarkdown are used to generate the final report.

Reticulate library (which handles Python in Rmarkdown files) needs to link
against libpython. So need to install a shared library with python, e.g.

```
$ env PYTHON_CONFIGURE_OPTS="--enabled-shared" pyenv install 3.7.3
```

Rtree, used by geopandas, also needs to link against a shared library called
libspatialjoin. This must be installed and available. If you install it in an
unusual place, you can always create an `.env` file and add:
```
LD_LIBRARY_PATH=/path/to/installed/library
```
