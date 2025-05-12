# Repository content

- `docs/`  
    - **Description**: Create documentation for all the apis in each package of EazyML
        - `_static`
        - `notebooks`
        - `packages`
        - `conf.py`
        - `index.rst`
        - `make.bat`
        - `Makefile`
- `packages/`  
    - **Description**: It contains client.py files for each package


# Generate Documentation

## Generate html page from .ipynb path
```bash
jupyter nbconvert --to html path_to_notebook.ipynb
```

## Steps to configure client files
1. copy respective package `client.py` file from `pip_eazyml` repository and paste it in packages directory here, with same parent directory name as in `pip_eazyml` repository.
2. Remove `@validate_license` decorator from all the client.py file.
3. update html and ipynb examples in `docs_eazyml\docs\_static` directory.

## Generate documentation using sphinx auto doc.
Enter below command in the base directory to get restructured text for specific module in docs directory.
```bash
sphinx-apidoc -o docs .\packages\augi\eazyml_insight\
```


## Generate html pages once every files are configured
    1. Go to `docs` directory
    2. Run below command to clean earlier `_build` directory
    ```bash
    make clean
    ```
    3. Run below command to generate `_build` directory where we have html pages.
    ```bash
    make html
    ```

# Generate Documentation in one step

1. Run below command to update documentation. Make sure `pip_eazyml` and `docs_eazyml` are in same directory and run below command from `pip_eazyml` directory.

    ```bash
    python update_documentation.py
    ```


