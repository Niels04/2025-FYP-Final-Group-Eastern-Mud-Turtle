from feature_extraction import extract
from codecarbon import track_emissions
from pathlib import Path

"""Helper python script to track the energy consumption for one run of feature extraction for each
of the models (baseline, extended, formula).
"""

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" #obtain data directory

# set up relevant directories
img_dir = str(_DATA_DIR / "lesion_imgs/")
mask_dir = str(_DATA_DIR / "lesion_masks/")
metadata_dir = str(_DATA_DIR / "metadata.csv")

@track_emissions(country_iso_code= 'DNK')
def emissions_baseline(img_dir, mask_dir, metadata_dir):
    return extract(img_dir, mask_dir, metadata_dir)

@track_emissions(country_iso_code= 'DNK')
def emissions_extended(img_dir, mask_dir, metadata_dir):
    return extract(img_dir, mask_dir, metadata_dir, base_model= False)

@track_emissions(country_iso_code= 'DNK')
def emissions_formula(img_dir, mask_dir, metadata_dir):
    return extract(img_dir, mask_dir, metadata_dir, formula_features= 'only')

emissions_baseline(img_dir, mask_dir, metadata_dir)
emissions_extended(img_dir, mask_dir, metadata_dir)
emissions_formula(img_dir, mask_dir, metadata_dir)


