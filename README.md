# Panel mapper

Script to map predicted Bragg spot centroids to their corresponding detector panel.
These panel ids can be used as additional features in learning algorithms. 

## Installation instructions

To use the panel mapper, 

```bash
# download the repo
git clone https://github.com/Hekstra-Lab/panel_mapper.git

# go to the repo 
cd panel_mapper

# create a conda environment 
conda env create -f env.yaml

# activate environment
conda activate mappanel

# install pnel mapper
pip install -e . 

```

To confirm the installation succeeded, run the following command

```bash 
map-refl-to-panel  --help
```
This should display the different arguments.

## Usage

```bash
map-refl-to-panel \
--x_det 3840 \ # number of pixels in horizontal axis
--y_det 3840 \ # number of pixels in vertical axis
--x_panel 16 \ # nuber of panels along horizontal axis
--y_panel 16 \ # number of panels along vertical axis
```
