Win

```bash
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
conda env create -f py37-torch113-cuda117_win.yml
conda activate libcity
pip install -r requirements_win.txt
```

Linux

```bash
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
conda env create -f py37-torch113-cuda117_linux.yml
```

