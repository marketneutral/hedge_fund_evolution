# TODO
- [X] Gitlab connection -- it just works; don't use Gitlab extension
- [ ] MS SQL connection
- [ ] Private pypi connection


# Admin Server Setup

- The Quarto CLI is not installed by default. An admin can install it with

```bash
- wget https://quarto.org/download/latest/quarto-linux-amd64.deb
- sudo dpkg -i quarto-linux-amd64.deb
```

# One Time Setup
- You need the Jupyter extention installed in your VS Code. Navigate to the Extensions pane and ensure it is installed. If it is the fist time, you need to also click to install it on the remote host from the Extensions pane.
- Get the Quarto extention

# Starting a New Project

- In Gitlab, ...
- File --> New Window --> Remote-SSH: Connect to Host --> pick server
- Click Clone Git Repository...
- paste in the repo URL


- Open a terminal (````Ctrl+` ````) 
- Create a new directory for your project

```bash
mkdir my_project
```
- Open the new directory in VS Code (File --> Open Folder)
- Make an `environment.yml` file with the following contents (can copy from another repo)
- Minimally, the yaml file should look like this

```yaml
name: my_project
channels:
  - defaults
dependencies:
  - python=3.12
  - boto3
  - jupyter
  - pandas
  - numpy
  - matplotlib
  - plotly
  - seaborn
  - statsmodels
  - scikit-learn
  - scipy
  - joblib
```


# Make the conda environment for this project

Now that you have the `environment.yml` file, you can make the conda environment (note, `mamba` is a faster version of the `conda` command).

```bash
mamba env create -f environment.yml
```

If you are updating, you can do `mamba env update --file environment.yml --prune`.

# Activate your conda environment (assume we called it `nb_test`).

```bash
conda activate nb_test
```

# Create notebook and choose kernel

Create an `.ipynb` file. On the top right of the window, choose the kernel (it should look like `~/.conda/envs/{env name}/bin/python`).


# Quarto

Note that if you have plotly plots in an `.ipynb`, you need to have
```python
import plotly.io as pio
pio.renderers.default = "notebook"
```
in the notebook to get the plots to show up in quarto.

