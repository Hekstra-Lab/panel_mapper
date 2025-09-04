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

By default, `panelmapper` assumes data were collected on a [Rayonix MX340-HS](https://biocars.uchicago.edu/facilities/experimental-station-equipment/) with binning set to 2x2.

```bash
map-refl-to-panel tests/data/subset.refl \
--x-det=3840 \
--y-det=3840 \
--x-panel=16 \
--y-panel=16 \
```
