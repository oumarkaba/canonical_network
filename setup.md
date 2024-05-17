# Setup Guide
**Note:** The following issues occured on the Windows Subsystem for Linux (WSL). I was succesful in following these instructions in the exact order listed below. Results may vary.

1. Start by installing `pytorch3d`. To do this, execute the following commands. I recommend copying  these instructions exactly, even if you already have `torchvision` and `cuda` installed, since `pytorch3d` is very picky about the versions of packages.
```{console}
conda create -n pytorch3d python=3.9
conda activate pytorch3d
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
```
Next, run the following.
```{console}
conda install pytorch3d -c pytorch3d
```

Once this is done, a conda environment will be created named `pytorch3d`; all the following commands were executed under this environment.

2. Next, you will possibly run into some of the following errors: 

    - `ModuleNotFoundError: No module named 'canonical_network'`

        To solve this, navigate to `/canonical_network` and run the command `pip install -e .`

    - `ERROR: Could not find a version that satisfies the requirement torch==1.10.1 (from canonical-network) (from versions: 1.13.0, 1.13.1, 2.0.0, 2.0.1, 2.1.0, 2.1.1, 2.1.2)
           ERROR: No matching distribution found for torch==1.10.1`

        To solve this, open pyproject.toml and change the torch version to the one currently installed.

    - `ERROR: Failed building wheel for torch-scatter`

        To solve this, delete line in pyproject.toml for torch-scatter

3. Next, install the required packages
```{console}
conda install lightning -c conda-forge
pip install wandb
conda install conda-forge::kornia
conda install conda-forge::torch-scatter
```

4. Finally, you need to upload file `canonical_network/canonical_network/data/n_body_system/dataset` (currently not on the GitHub repo.)

5. (Optional) Can rename environment.
```
conda rename -n pytorch3d new-env-name
```