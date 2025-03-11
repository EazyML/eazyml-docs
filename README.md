# Generate Documentation

1. Go to `docs` directory
2. Run below command to clean earlier `_build` directory
```bash
make clean
```
3. Run below command to generate `_build` directory where we have html pages.
```bash
make html
```

# Generate html page from .ipynb path
```bash
jupyter nbconvert --to html path_to_notebook.ipynb
```

# Steps to generate documentation 
1. copy respective package `client.py` file from `pip_eazyml` repository and paste it in packages directory here, with same parent directory name as in `pip_eazyml` repository.
2. Remove `@validate_license` decorator from all the client.py file.



