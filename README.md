# Visualize chain contacts in PyMOL

This script is used to visualize chain contacts in PyMOL.

## Usage

### Install PyMOL

On Mac or Linux machine, you can install PyMOL using conda:

```bash
# create a new environment if needed
conda create -n pymol python=3.12
conda activate pymol
# install PyMOL
conda install -c conda-forge pymol-open-source
# set alias in your shell configuration file
echo "alias pymol='$(which pymol)'" >> ~/.bashrc
```

### Install dependencies

```bash
conda activate pymol
pip install -r requirements.txt
```

### Install the plugin in PyMOL

Go to tab `Plugin` -> `Plugin Manager` -> `Install New Plugin` -> Under `Install from local file` click `Choose file ...` and select the script `cal_chain_dist_pymol.py`.
