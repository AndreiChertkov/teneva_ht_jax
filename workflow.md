# workflow

> Workflow instructions for `teneva_ht_jax` developers.


## How to install the current local version

1. Install [python](https://www.python.org) (version 3.8; you may use [anaconda](https://www.anaconda.com) package manager);

2. Create a virtual environment:
    ```bash
    conda create --name teneva_ht_jax python=3.8 -y
    ```

3. Activate the environment:
    ```bash
    conda activate teneva_ht_jax
    ```

4. Install special dependencies (for developers):
    ```bash
    pip install sphinx twine jupyterlab
    ```

5. Install `teneva_ht_jax`:
    ```bash
    python setup.py install
    ```

6. Reinstall locally `teneva_ht_jax` (after updates of the code):
    ```bash
    clear && pip uninstall teneva_ht_jax -y && python setup.py install
    ```

7. Rebuild the docs (after updates of the code):
    ```bash
    python doc/build.py
    ```

8. Delete virtual environment at the end of the work (optional):
    ```bash
    conda activate && conda remove --name teneva_ht_jax --all -y
    ```


## How to add a new function

1. Choose the most suitable module from `teneva_ht_jax` folder

2. Choose the name for function in lowercase

3. Add new function in alphabetical order, separating it with two empty lines from neighboring functions

4. Add function in alphabetical order into `__init__.py`

5. Make documentation (i.e., `docstring`) for the function similar to other functions

6. Prepare a demo (add it in alphabetical order) for the function in the related jupyter notebook in the `demo` folder (with the same name as a module name). Use the style similar to demos for other functions
    > Note that it's important to use a consistent style for all functions, as the code is then automatically exported from the jupyter notebooks to assemble the online documentation

7. Add new function name into dict in docs `doc/map.py` and rebuild the docs (run `python doc/build.py`), then check the result in web browser (see `doc/_build/html/index.html`)

8. Make commit

9. Use the new function locally until update of the package version


## How to update the package version

1. Update version (like `0.1.X`) in the file `teneva_ht_jax/__init__.py`
    > For breaking changes we should increase the major index (`1`), for non-breaking changes we should increase the minor index (`X`)

2. Build the docs `python doc/build.py`

3. Do commit `Update version (0.1.X)` and push it

4. Upload new version to `pypi` (login: `AndreiChertkov`; passw: `xxx`)
    ```bash
    rm -r ./dist && python setup.py sdist bdist_wheel && twine upload dist/*
    ```

5. Reinstall and check that installed version is the correct
    ```bash
    pip install --no-cache-dir --upgrade teneva_ht_jax
    ```

6. Check the [teneva_ht_jax docs build](https://readthedocs.org/projects/teneva_ht_jax/builds/)

7. Check the [teneva_ht_jax docs site](https://teneva-ht-jax.readthedocs.io/)
