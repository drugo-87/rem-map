import numpy as np
import math
import xarray as xr
import pandas as pd
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import (
    LinearSegmentedColormap,
    BoundaryNorm,
    ListedColormap,
    LogNorm,
)
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import os
import warnings

# warnings.simplefilter(action='ignore', category=FutureWarning)
###################################
############# Feeding #############
###################################

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--mode", help="Choose running modality ['archivio', 'prof_vert']", default="data/")
parser.add_argument("-d", "--date", help="Use bash command 'date +%Y%m%d_%H%M' ")
parser.add_argument("-i", "--input", required=False, help="the data-input folder", default="data/")
parser.add_argument("-o", "--output", required=False, help="the output folder", default="maps/")
# parser.print_help()
args = parser.parse_args()

running_modality = ['archivio', 'prof_vert']
if args.mode not in running_modality:
    print(f"{args.mode} is not present in the running modality")
    exit()

###################################
###################################
###################################

# Carica il file CSV

file_path = f'{args.input}/dati_{args.date}.csv' 
try:
    data = pd.read_csv(file_path,na_values="NA")
except:
    print("Something went wrong, can't find data file")
    exit()

data.columns = data.columns.str.strip()
data['datetime'] = pd.to_datetime(data['datetime'])
filtered_data = data[
    ~data['stazione'].str.contains('Camporella|Civago', na=False)
][['datetime', 'quota', 'temp_val', 'dewpoint', 'temp_wet_bulb']].dropna()
sorted_data = filtered_data[['datetime', 'quota', 'temp_val', 'dewpoint', 'temp_wet_bulb']].sort_values(by='quota', ascending=True)


fig = plt.figure(figsize=(12, 8))

if not sorted_data['temp_val'].isna().all():
    plt.plot(
        sorted_data['temp_val'].dropna(),
        sorted_data['quota'][sorted_data['temp_val'].notna()],
        label='Temperatura (°C)',
        color='red',
    )

if not sorted_data['dewpoint'].isna().all():
    plt.plot(
        sorted_data['dewpoint'].dropna(),
        sorted_data['quota'][sorted_data['dewpoint'].notna()],
        label='Punto di rugiada (°C)',
        color='green',
    )

if not sorted_data['temp_wet_bulb'].isna().all():
    plt.plot(
        sorted_data['temp_wet_bulb'].dropna(),
        sorted_data['quota'][sorted_data['temp_wet_bulb'].notna()],
        label='Bulbo umido (°C)',
        color='blue',
    )

plt.gca()
plt.ylim([0, 2100])

if args.mode == "archivio":
    month = sorted_data['datetime'].dt.month.iloc[0]
    if month in [12, 1, 2]:  # Inverno
        x_range = [-35, 20]
    elif month in [6, 7, 8]:  # Estate
        x_range = [-15, 40]
    else:  # Altri mesi
        x_range = [-25, 30]
    plt.xlim(x_range)

plt.xlabel('Gradi (°C)')
plt.ylabel('Quota (m)')
plt.title(f'Temperatura, Punto di rugiada e Bulbo Umido vs Quota ({sorted_data["datetime"].dt.strftime("%Y-%m-%d").iloc[0]})')
plt.legend()
plt.grid(True)

try:
    fig.savefig(
            f"{args.output}/{args.mode}/prof_vert_{args.date}.png",
            bbox_inches="tight",
            transparent=False,
        )
except:
    print("Can't save the figure, is the output path correct?")
    exit()
